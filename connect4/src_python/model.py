"""
model.py — Connect4Net: a compact Transformer with policy, value and
confidence heads.

Architecture
------------
Input: (B, 3, 6, 7) tensor. Three planes per cell:

    plane 0 = "own" pieces (the current player to move)
    plane 1 = "opponent" pieces
    plane 2 = turn indicator (all 1s — a constant bias, signals that
              the input is canonical to the side to move)

Trunk: a 3×3 conv lifts the 3 input planes to `channels` feature maps,
then `num_blocks` residual blocks. The residual block is the standard
two-conv + skip from ResNet, with BatchNorm. For Connect 4 the board is
so small that we don't need downsampling — the convs all use padding=1
to preserve the 6×7 spatial size.

Policy head: a 1×1 conv to 2 channels → flatten → Linear(2·6·7, 7) →
log_softmax over the 7 columns. log-probabilities are returned (not raw
logits) so the loss is a clean dot product with the MCTS target policy.

Value head: a 1×1 conv to 1 channel → flatten → Linear(1·6·7, 64) →
ReLU → Linear(64, 1) → tanh. Output is a scalar in [-1, +1] from the
perspective of the player to move at the input state.

Loss (computed in train.py, not here):

    L = (z - v)^2 - sum_a pi_a * log p_a + confidence_loss

The first term is the value MSE; the second is the cross-entropy between
the MCTS-improved policy pi and the network's policy p. Both reductions
are means over the batch. No L2 reg on the parameters — Adam with
weight_decay=1e-4 in train.py is enough to keep things tame.

Why this size?
--------------
The network is intentionally small for Connect 4 while retaining enough
capacity for policy, value and confidence calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from config import CONFIG
    _DEFAULT_D_MODEL = getattr(CONFIG.network, "d_model", 64)
    _DEFAULT_NUM_LAYERS = getattr(CONFIG.network, "num_layers", 4)
    _DEFAULT_NHEAD = getattr(CONFIG.network, "nhead", 4)
except Exception:
    _DEFAULT_D_MODEL = 64
    _DEFAULT_NUM_LAYERS = 4
    _DEFAULT_NHEAD = 4


# ---------------------------------------------------------------------------
# Connect4Net (Transformer)
# ---------------------------------------------------------------------------

class Connect4Net(nn.Module):
    """AlphaZero-style policy/value network with a learned confidence head.

    Args:
        d_model:   width of the transformer trunk (default 64).
        num_layers: number of transformer layers (default 4).
        nhead:     number of attention heads (default 4).
    """

    def __init__(self, d_model: int = 64, num_layers: int = 4, nhead: int = 4, **kwargs) -> None:
        # **kwargs catches old arguments like channels/num_blocks to prevent crashes
        super().__init__()
        self.d_model = d_model
        self.channels = d_model
        self.num_layers = num_layers
        self.num_blocks = num_layers
        self.nhead = nhead

        # Flattened board: 6x7 = 42 squares. Each square starts with 3 features -> d_model
        self.token_proj = nn.Linear(3, d_model)
        
        # Positional Embedding for the 42 squares
        self.pos_emb = nn.Parameter(torch.randn(1, 42, d_model) * 0.02)
        
        # Transformer Trunk
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Policy head: takes (B, d_model, 6, 7) -> 1x1 conv to 2 channels -> flatten -> Linear(2*6*7, 7)
        self.policy_conv = nn.Conv2d(d_model, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)

        # Value head
        self.value_conv = nn.Conv2d(d_model, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Moves Left head (Auxiliary)
        self.moves_left_conv = nn.Conv2d(d_model, 1, kernel_size=1, bias=False)
        self.moves_left_bn = nn.BatchNorm2d(1)
        self.moves_left_fc1 = nn.Linear(1 * 6 * 7, 64)
        self.moves_left_fc2 = nn.Linear(64, 1)

        # Predict how concentrated the normal PCR self-play MCTS target is.
        # This is a learned confidence signal, not a tactical board rule.
        self.confidence_conv = nn.Conv2d(d_model, 1, kernel_size=1, bias=False)
        self.confidence_bn = nn.BatchNorm2d(1)
        self.confidence_fc1 = nn.Linear(1 * 6 * 7, 32)
        self.confidence_fc2 = nn.Linear(32, 1)
        self.has_confidence_head = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the network.

        Args:
            x: (B, 3, 6, 7) input tensor (own / opponent / turn planes).

        Returns:
            log_p: (B, 7) log-probabilities over the 7 columns.
            v:     (B,)   predicted value in [-1, +1].
            m:     (B,)   predicted moves left.
            c:     (B,)   predicted confidence in the selected move, [0, 1].
        """
        B = x.size(0)
        
        # Flatten spatial dims to sequence of 42 tokens: (B, 3, 6, 7) -> (B, 3, 42) -> (B, 42, 3)
        tokens = x.view(B, 3, 42).permute(0, 2, 1)
        
        # Project tokens and add positional embedding
        h = self.token_proj(tokens) + self.pos_emb # (B, 42, d_model)
        
        # Transformer layers
        h = self.transformer(h)
        h = self.norm(h)
        
        # Reshape back to spatial tensor: (B, 42, d_model) -> (B, d_model, 42) -> (B, d_model, 6, 7)
        h_spatial = h.permute(0, 2, 1).view(B, self.d_model, 6, 7).contiguous()
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h_spatial)), inplace=True)
        p = p.flatten(start_dim=1)
        logits = self.policy_fc(p)
        
        # --- ACTION MASKING ---
        # x[:, 0, 5, :] = own pieces at top row (r=5)
        # x[:, 1, 5, :] = opp pieces at top row (r=5)
        top_row_occupied = (x[:, 0, 5, :] + x[:, 1, 5, :]) > 0.5
        logits = logits.masked_fill(top_row_occupied, -1e9)
        log_p = F.log_softmax(logits, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(h_spatial)), inplace=True)
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = torch.tanh(self.value_fc2(v)).squeeze(1)
        
        # Moves Left head
        m = F.relu(self.moves_left_bn(self.moves_left_conv(h_spatial)), inplace=True)
        m = m.flatten(start_dim=1)
        m = F.relu(self.moves_left_fc1(m), inplace=True)
        # We use ReLU at the very end to ensure moves left is >= 0
        m = F.relu(self.moves_left_fc2(m)).squeeze(1)

        c = F.relu(self.confidence_bn(self.confidence_conv(h_spatial)), inplace=True)
        c = c.flatten(start_dim=1)
        c = F.relu(self.confidence_fc1(c), inplace=True)
        c = torch.sigmoid(self.confidence_fc2(c)).squeeze(1)

        return log_p, v, m, c

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), tmp_path)
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> Connect4Net:
        d_model = kwargs.get("d_model", _DEFAULT_D_MODEL)
        num_layers = kwargs.get("num_layers", _DEFAULT_NUM_LAYERS)
        nhead = kwargs.get("nhead", _DEFAULT_NHEAD)
        net = cls(d_model=d_model, num_layers=num_layers, nhead=nhead)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        missing, _unexpected = net.load_state_dict(state_dict, strict=False)
        net.has_confidence_head = not any(
            name.startswith("confidence_") for name in missing
        )
        return net
