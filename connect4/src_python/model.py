"""
model.py — Connect4Net: a small ResNet-style CNN with policy + value heads.

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

    L = (z - v)^2 - sum_a pi_a * log p_a

The first term is the value MSE; the second is the cross-entropy between
the MCTS-improved policy pi and the network's policy p. Both reductions
are means over the batch. No L2 reg on the parameters — Adam with
weight_decay=1e-4 in train.py is enough to keep things tame.

Why this size?
--------------
~50K parameters. Trains in seconds on a 1650, fits in L2 with room to
spare. For Connect 4 you don't need a bigger model — the state space is
bounded and the patterns are local (3×3 convs cover any potential).
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
    _DEFAULT_CHANNELS = CONFIG.network.channels
    _DEFAULT_NUM_BLOCKS = CONFIG.network.num_blocks
except Exception as err:
    print(f"[model] WARNING: Failed to load config.py ({err}); using fallbacks (channels=64, num_blocks=3)")
    _DEFAULT_CHANNELS = 64
    _DEFAULT_NUM_BLOCKS = 3


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two 3×3 convs + skip connection, with BatchNorm. Preserves shape."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + x, inplace=True)


# ---------------------------------------------------------------------------
# Connect4Net
# ---------------------------------------------------------------------------

class Connect4Net(nn.Module):
    """AlphaZero-style policy + value network for Connect 4.

    Args:
        channels:  width of the conv trunk.
        num_blocks: number of residual blocks in the trunk.
    """

    def __init__(
        self,
        channels: int = _DEFAULT_CHANNELS,
        num_blocks: int = _DEFAULT_NUM_BLOCKS,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks

        # Lift 3 input planes -> `channels` feature maps.
        self.input_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual trunk.
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head: 1×1 conv to 2 channels -> flatten -> Linear -> log_softmax.
        # 2 channels is the AlphaZero convention; it gives the head a tiny
        # bit of expressivity before the linear projection.
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)

        # Value head: 1×1 conv to 1 channel -> flatten -> 64-d hidden -> 1.
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network.

        Args:
            x: (B, 3, 6, 7) input tensor (own / opponent / turn planes).

        Returns:
            log_p: (B, 7) log-probabilities over the 7 columns.
            v:     (B,)   predicted value in [-1, +1] per sample.
        """
        # Trunk.
        h = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
        for block in self.blocks:
            h = block(h)

        # Policy head.
        p = F.relu(self.policy_bn(self.policy_conv(h)), inplace=True)
        p = p.reshape(p.size(0), -1)  # (B, 84)
        p = self.policy_fc(p)         # (B, 7) — raw logits
        log_p = F.log_softmax(p, dim=1)

        # Value head. We use .reshape (not .squeeze) to drop the trailing
        # size-1 dimension. Empirically onnxruntime mis-handles .squeeze
        # on a (B, 1) tensor when the batch dim is dynamic — the value
        # output collapses to shape (1,) instead of (B,). .reshape is a
        # static shape op that ORT handles correctly.
        v = F.relu(self.value_bn(self.value_conv(h)), inplace=True)
        v = v.reshape(v.size(0), -1)  # (B, 42)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = torch.tanh(self.value_fc2(v))             # (B, 1)
        v = v.reshape(v.size(0))                      # (B,)

        return log_p, v

    # -- I/O helpers --------------------------------------------------------

    def save(self, path: str) -> None:
        """Save state_dict to `path`."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str | torch.device = "cpu") -> None:
        """Load state_dict from `path`. Tolerates a `model.` prefix on keys
        (e.g. if the file was saved from a `torch.compile`d module)."""
        sd = torch.load(path, map_location=map_location)
        if any(k.startswith("model.") for k in sd.keys()):
            sd = {k.removeprefix("model."): v for k, v in sd.items()}
        self.load_state_dict(sd)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Quick smoke test — `python model.py` will print a forward pass.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    net = Connect4Net()
    print(f"Connect4Net: {net.num_parameters():,} parameters")
    x = torch.randn(4, 3, 6, 7)
    log_p, v = net(x)
    print(f"input  : {tuple(x.shape)}")
    print(f"log_p  : {tuple(log_p.shape)}  (log-probs, sums to ~0)")
    print(f"v      : {tuple(v.shape)}  range [{v.min().item():.3f}, {v.max().item():.3f}]")
    print(f"log_p  : {log_p[0].exp()}  (sample policy, sums to {log_p[0].exp().sum().item():.4f})")
