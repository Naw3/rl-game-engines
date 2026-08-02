"""
train.py — Train Connect4Net on a C4D1 self-play dataset, then export
the trained model to ONNX for the Rust MCTS to consume.

Loss
----
The total loss is a sum of two terms (both reduced as means over the batch):

    L_total  = L_value + L_policy
    L_value  = (z - v)^2                                       (MSE)
    L_policy = - sum_a  pi_a * log p_a                         (cross-entropy)

where for each sample:
    z     ∈ {-1, 0, +1}   is the game outcome from the perspective
                          of the player to move at this state.
    v     ∈ [-1, +1]      is the value head's prediction.
    pi    ∈ Δ^6           is the MCTS-improved policy (length 7,
                          sum-to-1 over legal moves, zero elsewhere).
    p     ∈ Δ^6           is the network's policy (length 7, sum-to-1
                          after softmax — we keep log p as the output
                          of the head, see `model.py`).

The cross-entropy is `−π · log p`, which is the standard AlphaZero
formulation: it equals KL(π ∥ p) + H(π), and we drop the constant
H(π) by definition (the target distribution doesn't depend on the
network). Maximising log-likelihood of the MCTS policy ≡ minimising
this term.

After every cycle
-----------------
The trained `state_dict` is saved to `connect4_model.pt` AND exported
to `connect4_model.onnx` (consumed by the Rust MCTS in the next
self-play cycle). The ONNX export uses dynamic batch dim, named I/O
("input" / "policy" / "value"), and opset 17. tract-onnx 0.21 reads
it without any conversion.

Numerical notes
---------------
* Mixed precision (AMP FP16) is used on CUDA. On Turing/Volta (sm_75,
  e.g. the 1650) FP16 is the only native half-precision — BF16 is not
  supported. The model is small and the target distributions are
  well-behaved, so FP16 + GradScaler is stable. If you see NaN losses,
  try `--no-amp` to confirm.
* torch.compile() is applied by default. The first epoch is slow
  (compilation); subsequent epochs are 2–4× faster on small models.
  Disable with `--no-compile` if your PyTorch version is <2.0 or if
  you hit Triton/CUDA issues.

CLI
---
    python train.py --data selfplay.bin --out connect4_model.pt
                    [--epochs 5] [--batch 256] [--lr 1e-3]
                    [--no-amp] [--no-compile] [--no-onnx]
                    [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import re
import sys
import time
from pathlib import Path
import warnings
import logging

# Force UTF-8 on stdout/stderr. Windows defaults to cp1252 which crashes on
# Unicode arrows / Greek letters in print() output. Safe on all platforms.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Suppress standard Python warnings and specific library warnings
warnings.filterwarnings("ignore")

# PyTorch 2.5+ uses standard python logging for torch.onnx
# We set the level to ERROR to suppress the verbose export logs.
logging.getLogger("torch.onnx").setLevel(logging.ERROR)


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) for PyTorch model weights."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])
logging.getLogger("torch.onnx._internal").setLevel(logging.ERROR)
logging.getLogger("torch.export").setLevel(logging.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import CONFIG
    _DEFAULT_DATA = str(CONFIG.paths.selfplay_bin.name)
    _DEFAULT_OUT = str(CONFIG.paths.model_pt)
    _DEFAULT_EPOCHS = CONFIG.train.epochs
    _DEFAULT_BATCH = CONFIG.train.batch_size
    _DEFAULT_LR = CONFIG.train.learning_rate
    _DEFAULT_LR_MIN = CONFIG.train.learning_rate_min
    _DEFAULT_LR_WARMUP_EPOCHS = CONFIG.train.lr_warmup_epochs
    _DEFAULT_LR_SCHEDULE_EPOCHS = CONFIG.train.lr_schedule_epochs
    _DEFAULT_CONFIDENCE_LOSS_WEIGHT = CONFIG.train.confidence_loss_weight
    _DEFAULT_WD = CONFIG.train.weight_decay
    _DEFAULT_SEED = CONFIG.mcts.seed
    _DEFAULT_NUM_WORKERS = CONFIG.train.num_workers
    _DEFAULT_LOG_EVERY = CONFIG.train.log_every
    _DEFAULT_ONNX_EVERY = CONFIG.train.onnx_every
    _DEFAULT_USE_EMA = CONFIG.train.use_ema
    _DEFAULT_EMA_DECAY = CONFIG.train.ema_decay
    _DEFAULT_OPSET = CONFIG.dataset.onnx_opset
    _DEFAULT_PLANES = CONFIG.network.input_planes
    _DEFAULT_ROWS = CONFIG.network.board_rows
    _DEFAULT_COLS = CONFIG.network.board_cols
    _DEFAULT_MAX_ONNX_BATCH = CONFIG.dataset.max_onnx_batch
    _DEFAULT_MAX_GRAD_NORM = CONFIG.train.max_grad_norm
    _DEFAULT_SYMMETRY = CONFIG.train.symmetry
    _DEFAULT_NO_AMP = CONFIG.train.train_precision == "fp32"
    _DEFAULT_NO_COMPILE = CONFIG.train.compile_mode == "none"
    _DEFAULT_COMPILE_MODE = CONFIG.train.compile_mode
    _DEFAULT_INFER_PRECISION = CONFIG.train.infer_precision
    _DEFAULT_CHANNELS_LAST = CONFIG.train.channels_last
    _DEFAULT_FUSED_ADAMW = CONFIG.train.fused_adamw
except Exception as err:
    print(f"[train] WARNING: Failed to load config.py ({err}); using fallbacks")
    _DEFAULT_DATA = "selfplay.bin"
    _DEFAULT_OUT = str(_PROJECT_ROOT / "models" / "connect4_model.pt")
    _DEFAULT_EPOCHS = 5
    _DEFAULT_BATCH = 256
    _DEFAULT_LR = 1e-3
    _DEFAULT_LR_MIN = 1e-5
    _DEFAULT_LR_WARMUP_EPOCHS = 5
    _DEFAULT_LR_SCHEDULE_EPOCHS = 400
    _DEFAULT_CONFIDENCE_LOSS_WEIGHT = 0.1
    _DEFAULT_WD = 1e-4
    _DEFAULT_SEED = 42
    _DEFAULT_NUM_WORKERS = 0
    _DEFAULT_LOG_EVERY = 20
    _DEFAULT_ONNX_EVERY = 0
    _DEFAULT_USE_EMA = True
    _DEFAULT_EMA_DECAY = 0.999
    _DEFAULT_OPSET = 18
    _DEFAULT_PLANES = 3
    _DEFAULT_ROWS = 6
    _DEFAULT_COLS = 7
    _DEFAULT_MAX_ONNX_BATCH = 256
    _DEFAULT_MAX_GRAD_NORM = 5.0
    _DEFAULT_SYMMETRY = True
    _DEFAULT_NO_AMP = False
    _DEFAULT_NO_COMPILE = False
    _DEFAULT_COMPILE_MODE = "reduce-overhead"
    _DEFAULT_INFER_PRECISION = "fp32"
    _DEFAULT_CHANNELS_LAST = False
    _DEFAULT_FUSED_ADAMW = False


_EPOCH_CHECKPOINT_RE = re.compile(
    r"^(?P<base>.+?)_epoch(?P<epoch>\d+)(?P<ema>_ema)?$",
    re.IGNORECASE,
)


def format_duration(seconds: float) -> str:
    """Format short durations without losing sub-second timings."""
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        milliseconds = 0 if seconds == 0.0 else max(1, int(round(seconds * 1000.0)))
        return f"{milliseconds}ms"
    return f"{seconds:.2f}s"


def checkpoint_epoch(checkpoint_path: str | Path) -> int:
    """Read the cumulative epoch encoded in a checkpoint filename."""
    match = _EPOCH_CHECKPOINT_RE.match(Path(checkpoint_path).stem)
    return int(match.group("epoch")) if match else 0


def checkpoint_path_for_epoch(checkpoint_path: str | Path, epoch: int) -> Path:
    """Build a new checkpoint filename containing the cumulative epoch."""
    path = Path(checkpoint_path)
    match = _EPOCH_CHECKPOINT_RE.match(path.stem)
    if match:
        stem = f"{match.group('base')}_epoch{epoch}{match.group('ema') or ''}"
    else:
        stem = f"{path.stem}_epoch{epoch}"
    return path.with_name(stem + path.suffix)


def latest_epoch_checkpoint(checkpoint_path: str | Path) -> Path | None:
    """Find the newest raw checkpoint beside a stable model path."""
    path = Path(checkpoint_path)
    base_stem = path.stem
    current = _EPOCH_CHECKPOINT_RE.match(base_stem)
    if current:
        base_stem = current.group("base")

    candidates: list[tuple[int, Path]] = []
    for candidate in path.parent.glob(f"{base_stem}_epoch*.pt"):
        match = _EPOCH_CHECKPOINT_RE.match(candidate.stem)
        if match and match.group("base") == base_stem and not match.group("ema"):
            candidates.append((int(match.group("epoch")), candidate))

    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def load_training_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device,
) -> tuple[dict, dict | None]:
    """Load a model checkpoint and optionally return its training state.

    Stable ``connect4_model.pt`` files intentionally stay plain state dicts so
    Rust/ONNX tooling can consume them.  Cumulative ``*_epochN.pt`` files may
    additionally contain optimizer, scheduler, scaler and EMA state.
    """
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        return payload["model_state_dict"], payload
    return payload, None


def _copy_to_cpu(value):
    """Recursively copy tensors in a training state to CPU for portable saves."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return value


def save_training_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: ExponentialMovingAverage | None,
    scaler,
    best_loss: float,
) -> None:
    """Save all state needed to continue training without resetting dynamics."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": _copy_to_cpu(model.state_dict()),
        "optimizer_state_dict": _copy_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": _copy_to_cpu(scheduler.state_dict()),
        "ema_shadow": _copy_to_cpu(ema.shadow) if ema is not None else None,
        "scaler_state_dict": _copy_to_cpu(scaler.state_dict()),
        "best_loss": float(best_loss),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast, GradScaler
except ImportError:  # PyTorch < 2.0
    from torch.cuda.amp import autocast, GradScaler

from dataset import C4Dataset
from model import Connect4Net
from utils.onnx_export import (
    export_onnx,
    export_onnx_to_bytes,
    update_onnx_weights_bytes,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    log_p: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    target_moves_left: torch.Tensor,
    predicted_confidence: torch.Tensor | None = None,
    confidence_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, policy_loss, value_loss, moves_left_loss).

    The confidence target is the concentration of the MCTS visit policy.  It
    is trained even when confidence-based early stopping is disabled, so the
    inference log remains useful in fixed-time mode too.
    """
    # Policy: cross-entropy with soft targets. target_policy is (B, 7),
    # log_p is (B, 7) of log-probabilities. Sum over columns, mean over batch.
    policy_loss = -(target_policy * log_p).sum(dim=1).mean()
    # Value: standard MSE.
    value_loss = F.mse_loss(v, target_value)
    # Moves left: standard MSE, scaled down to prevent dominating the gradients.
    moves_left_loss = F.mse_loss(m, target_moves_left) * 0.02
    confidence_loss = torch.zeros_like(policy_loss)
    if predicted_confidence is not None and confidence_loss_weight > 0.0:
        # MCTS policies are normalized distributions, so their maximum visit
        # probability is a direct, stable concentration/confidence target.
        confidence_target = target_policy.detach().amax(dim=1).clamp(0.0, 1.0)
        confidence_loss = F.mse_loss(predicted_confidence, confidence_target)
        confidence_loss = confidence_loss * confidence_loss_weight
    total_loss = policy_loss + value_loss + moves_left_loss + confidence_loss
    return total_loss, policy_loss, value_loss, moves_left_loss




class _ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, planes, policy, value, moves_left):
        self._planes, self._policy, self._value, self._moves_left = planes, policy, value, moves_left
        self.count = len(planes)
        self.symmetry = False
    def __len__(self): return self.count
    def __getitem__(self, idx):
        planes = self._planes[idx]
        policy = self._policy[idx]
        value = self._value[idx]
        moves_left = self._moves_left[idx]
        if self.symmetry:
            if random.random() < 0.5:
                planes = planes[:, :, ::-1].copy()
                policy = policy[::-1].copy()
        return (torch.from_numpy(planes),
                torch.from_numpy(policy),
                torch.tensor(value, dtype=torch.float32),
                torch.tensor(moves_left, dtype=torch.float32))


def merge_replay_datasets(
    datasets: list[_ReplayDataset],
    symmetry: bool,
) -> _ReplayDataset:
    """Merge recent self-play generations into one RAM training dataset."""
    if len(datasets) == 1:
        return datasets[0]

    import numpy as np

    merged = _ReplayDataset(
        np.concatenate([dataset._planes for dataset in datasets], axis=0),
        np.concatenate([dataset._policy for dataset in datasets], axis=0),
        np.concatenate([dataset._value for dataset in datasets], axis=0),
        np.concatenate([dataset._moves_left for dataset in datasets], axis=0),
    )
    merged.symmetry = symmetry
    merged.n_games = sum(getattr(dataset, "n_games", 0) for dataset in datasets)
    return merged


def merge_weighted_replay_dataset(
    current: _ReplayDataset,
    history: list[_ReplayDataset],
    symmetry: bool,
) -> _ReplayDataset:
    """Keep one epoch-sized dataset with fresh and replay positions mixed.

    The replay window is only a source of samples.  It must not increase the
    number of optimizer batches for the epoch, otherwise ``replay_keep=10``
    would silently turn one self-play generation into ten generations worth
    of training data.  Keep the output exactly as large as the newest
    generation, with a 50/50 fresh/replay mix once replay is available.
    """
    if not history:
        return current

    import numpy as np

    old_arrays = history[:-1]
    if not old_arrays:
        return current

    old_planes = np.concatenate([dataset._planes for dataset in old_arrays], axis=0)
    old_policy = np.concatenate([dataset._policy for dataset in old_arrays], axis=0)
    old_value = np.concatenate([dataset._value for dataset in old_arrays], axis=0)
    old_moves_left = np.concatenate(
        [dataset._moves_left for dataset in old_arrays], axis=0
    )
    target_count = len(current)
    if target_count == 0:
        return current

    old_count = len(old_planes)
    replay_count = target_count // 2
    fresh_count = target_count - replay_count

    fresh_indices = np.random.choice(
        target_count, size=fresh_count, replace=False
    )
    replay_indices = np.random.choice(
        old_count, size=replay_count, replace=old_count < replay_count
    )

    planes = np.concatenate(
        [current._planes[fresh_indices], old_planes[replay_indices]], axis=0
    )
    policy = np.concatenate(
        [current._policy[fresh_indices], old_policy[replay_indices]], axis=0
    )
    value = np.concatenate(
        [current._value[fresh_indices], old_value[replay_indices]], axis=0
    )
    moves_left = np.concatenate(
        [current._moves_left[fresh_indices], old_moves_left[replay_indices]], axis=0
    )
    order = np.random.permutation(target_count)
    merged = _ReplayDataset(
        planes[order].copy(),
        policy[order].copy(),
        value[order].copy(),
        moves_left[order].copy(),
    )
    merged.symmetry = symmetry
    # This is still one fresh self-play generation from the pipeline's point
    # of view; replay positions do not create extra games or extra batches.
    merged.n_games = getattr(current, "n_games", 0)
    return merged


class WarmupCosineScheduler:
    """Small serializable warmup + cosine scheduler for online self-play."""

    scheduler_type = "warmup_cosine_v1"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        eta_min: float,
        start_lrs: list[float] | None = None,
        peak_lrs: list[float] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps))
        self.eta_min = max(0.0, float(eta_min))
        defaults = [float(group["lr"]) for group in optimizer.param_groups]
        self.peak_lrs = list(peak_lrs or defaults)
        self.start_lrs = list(start_lrs or [max(self.eta_min, lr * 0.1) for lr in self.peak_lrs])
        self.last_epoch = 0
        self._last_lr = list(self.start_lrs)
        self._set_lrs(self._last_lr)

    def _set_lrs(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = float(lr)
        self._last_lr = [float(lr) for lr in lrs]

    def _lrs_at(self, step: int) -> list[float]:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            fraction = step / float(self.warmup_steps)
            return [
                start + (peak - start) * fraction
                for start, peak in zip(self.start_lrs, self.peak_lrs)
            ]

        cosine_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / cosine_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (peak - self.eta_min) * cosine
            for peak in self.peak_lrs
        ]

    def step(self) -> None:
        self.last_epoch += 1
        self._set_lrs(self._lrs_at(self.last_epoch))

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def reconfigure_resume(
        self,
        start_lrs: list[float],
        peak_lrs: list[float],
        total_steps: int,
        warmup_steps: int,
    ) -> None:
        """Continue an old scheduler with a gradual LR recovery, not a reset."""
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, min(int(warmup_steps), self.total_steps))
        self.start_lrs = list(start_lrs)
        self.peak_lrs = list(peak_lrs)
        self.last_epoch = 0
        self._set_lrs(self.start_lrs)

    def state_dict(self) -> dict:
        return {
            "scheduler_type": self.scheduler_type,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "eta_min": self.eta_min,
            "start_lrs": self.start_lrs,
            "peak_lrs": self.peak_lrs,
            "last_epoch": self.last_epoch,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.total_steps = max(1, int(state_dict["total_steps"]))
        self.warmup_steps = max(0, min(int(state_dict["warmup_steps"]), self.total_steps))
        self.eta_min = max(0.0, float(state_dict["eta_min"]))
        self.start_lrs = [float(lr) for lr in state_dict["start_lrs"]]
        self.peak_lrs = [float(lr) for lr in state_dict["peak_lrs"]]
        self.last_epoch = int(state_dict["last_epoch"])
        self._set_lrs([float(lr) for lr in state_dict["_last_lr"]])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------



class CudaPrefetcher:
    def __init__(self, loader, device, channels_last=False):
        self.loader = iter(loader)
        self.device = device
        self.channels_last = channels_last
        self.use_cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.use_cuda else None
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
            
        if not self.use_cuda:
            planes = self.next_batch[0].to(self.device, non_blocking=True)
            if self.channels_last:
                planes = planes.to(memory_format=torch.channels_last)
            self.next_batch = (
                planes,
                self.next_batch[1].to(self.device, non_blocking=True),
                self.next_batch[2].to(self.device, non_blocking=True),
                self.next_batch[3].to(self.device, non_blocking=True),
            )
            return

        with torch.cuda.stream(self.stream):
            planes = self.next_batch[0].to(self.device, non_blocking=True)
            if self.channels_last:
                planes = planes.to(memory_format=torch.channels_last)
            self.next_batch = (
                planes,
                self.next_batch[1].to(self.device, non_blocking=True),
                self.next_batch[2].to(self.device, non_blocking=True),
                self.next_batch[3].to(self.device, non_blocking=True)
            )

    def next(self):
        if not self.use_cuda:
            batch = self.next_batch
            if batch is not None:
                self.preload()
            return batch

        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch[0].record_stream(torch.cuda.current_stream())
            batch[1].record_stream(torch.cuda.current_stream())
            batch[2].record_stream(torch.cuda.current_stream())
            batch[3].record_stream(torch.cuda.current_stream())
            self.preload()
        return batch

def main() -> None:
    p = argparse.ArgumentParser(description="Train Connect4Net on C4D1 data")
    p.add_argument("--data", default=_DEFAULT_DATA,
                   help="path to a single C4D1 file")
    p.add_argument("--data-dir", default=None,
                   help="path to a directory of C4D1 files (replay buffer); "
                        "all *.bin files in the dir are loaded and concatenated. "
                        "Overrides --data if set.")
    p.add_argument("--out", default=_DEFAULT_OUT, help="output model path")
    p.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--duration", type=int, default=None, help="Train for a specific duration in seconds")
    p.add_argument("--infer-precision", choices=["fp32", "fp16", "int8"], default=_DEFAULT_INFER_PRECISION, help="Precision for the exported ONNX model")
    p.add_argument("--compile-mode", type=str, default=_DEFAULT_COMPILE_MODE, choices=["none", "default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    p.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=_DEFAULT_LR)
    p.add_argument("--lr-min", type=float, default=_DEFAULT_LR_MIN,
                   help="minimum learning rate for the cosine schedule")
    p.add_argument("--lr-warmup-epochs", type=int, default=_DEFAULT_LR_WARMUP_EPOCHS,
                   help="linear learning-rate warmup duration")
    p.add_argument("--lr-schedule-epochs", type=int, default=_DEFAULT_LR_SCHEDULE_EPOCHS,
                   help="total epoch horizon used by the cosine schedule")
    p.add_argument("--confidence-loss-weight", type=float,
                   default=_DEFAULT_CONFIDENCE_LOSS_WEIGHT,
                   help="weight for the MCTS policy-concentration confidence loss")
    p.add_argument("--weight-decay", type=float, default=_DEFAULT_WD)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="compute device (auto = cuda if available else cpu)",
    )
    # Convenience flags that override --device.
    p.add_argument("--cpu", action="store_const", const="cpu", dest="device",
                   help="force CPU training (overrides --device)")
    p.add_argument("--gpu", action="store_const", const="cuda", dest="device",
                   help="force CUDA training (overrides --device)")
    p.add_argument("--no-amp", action="store_true", default=_DEFAULT_NO_AMP, help="disable FP16 autocast")
    p.add_argument("--no-compile", action="store_true", default=_DEFAULT_NO_COMPILE, help="disable torch.compile")
    p.add_argument("--no-onnx", action="store_true",
                   help="skip ONNX export (debug only)")
    p.add_argument("--onnx-opset", type=int, default=_DEFAULT_OPSET)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap dataset size (for quick smoke tests)")
    p.add_argument("--log-every", type=int, default=_DEFAULT_LOG_EVERY)
    p.add_argument("--onnx-every", type=int, default=_DEFAULT_ONNX_EVERY, help="Frequency (in epochs) to export ONNX model (0 = export only at final epoch)")
    p.add_argument("--use-ema", action="store_true", default=_DEFAULT_USE_EMA, help="enable Exponential Moving Average (EMA) of model weights")
    p.add_argument("--ema-decay", type=float, default=_DEFAULT_EMA_DECAY, help="EMA decay rate (default: 0.999)")
    p.add_argument("--replay-keep", type=int, default=getattr(CONFIG.train, "replay_keep", 10),
                   help="number of recent streaming self-play generations kept in RAM")
    p.add_argument("--channels-last", action="store_true", default=_DEFAULT_CHANNELS_LAST, help="use channels_last memory format")
    p.add_argument("--fused-adamw", action="store_true", default=_DEFAULT_FUSED_ADAMW, help="use fused AdamW optimizer")
    p.add_argument("--symmetry", action="store_true", default=_DEFAULT_SYMMETRY,
                   help="enable horizontal-flip augmentation (doubles effective dataset size)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: 2 for single file, 0 for replay "
                        "because the replay dataset is defined in main() and can't be pickled).")
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="RNG seed for training")
    args = p.parse_args()

    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.compile_mode == "none":
        args.no_compile = True

    # Resolve `--device auto` to a concrete device based on CUDA availability.
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[train] --device auto resolved to {args.device} "
              f"(torch.cuda.is_available()={torch.cuda.is_available()})")
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"[train] WARNING: --device cuda requested but torch.cuda.is_available() is False; "
              f"falling back to CPU")
        args.device = "cpu"

    print(f"[train] device={args.device}  amp={'off' if args.no_amp else 'on'}  "
          f"compile={'off' if args.no_compile else 'on'}  "
          f"onnx={'off' if args.no_onnx else 'on'}")
    print(f"[train] data={args.data}  out={args.out}")
    print(f"[train] epochs={args.epochs}  batch={args.batch}  lr={args.lr}")

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("[train] enabled cudnn.benchmark for faster convolutions")

    # ---- Model ------------------------------------------------------------
    model = Connect4Net().to(args.device)
    out_path = Path(args.out)
    resume_path = out_path
    if checkpoint_epoch(out_path) == 0:
        resume_path = latest_epoch_checkpoint(out_path) or out_path
    start_epoch = checkpoint_epoch(resume_path) if resume_path.exists() else 0
    resume_training_state = None
    if resume_path.exists():
        if resume_path != out_path:
            print(f"[train] Auto-selected newest checkpoint {resume_path.name} for resume.")
        print(f"[train] Loading existing weights from {resume_path} to resume training...")
        try:
            state_dict, resume_training_state = load_training_checkpoint(
                resume_path, map_location=args.device
            )
            model.load_state_dict(state_dict)
            print("[train] Successfully restored model weights!")
            if resume_training_state is None and start_epoch > 0:
                print(
                    "[train] WARNING: This legacy epoch checkpoint contains weights only; "
                    "optimizer/scheduler/EMA state cannot be restored."
                )
            if start_epoch > 0:
                print(
                    f"[train] Resuming from epoch {start_epoch}; "
                    f"this run targets epoch {start_epoch + args.epochs}."
                )
        except Exception as e:
            print(f"[train] WARNING: Could not load existing weights from {resume_path} ({e}); starting with fresh weights")
            start_epoch = 0
            resume_training_state = None

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = model.num_parameters()
    print(f"[train] model: {n_params:,} parameters")

    # ---- Stream Reader Helper (Read Rust MCTS stdout directly into RAM) ----
    def read_stream_batch(stream):
        import struct
        header = stream.read(16)
        if not header or len(header) < 16:
            return None, None
        magic = header[:4]
        if magic != b"C4D1":
            return None, None
        (count, n_games) = struct.unpack("<II", header[4:12])
        if count == 0:
            return None, None
        
        n_bytes = count * 60
        buf = bytearray()
        while len(buf) < n_bytes:
            chunk = stream.read(n_bytes - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        if len(buf) < n_bytes:
            return None, None
        
        import numpy as np
        from dataset import decode_bitboard_batched
        # Parse 60-byte structs from RAM buffer
        raw_arr = np.frombuffer(buf, dtype=np.uint8).reshape(count, 60)
        own_arr = raw_arr[:, 0:8].view(np.uint64).reshape(count)
        opp_arr = raw_arr[:, 8:16].view(np.uint64).reshape(count)
        policy_arr = raw_arr[:, 24:52].view(np.float32).reshape(count, 7)
        value_arr = raw_arr[:, 52:56].view(np.float32).reshape(count)
        moves_left_arr = raw_arr[:, 56:60].view(np.float32).reshape(count)

        planes_arr = decode_bitboard_batched(own_arr, opp_arr)
        ds = _ReplayDataset(planes_arr, policy_arr, value_arr, moves_left_arr)
        ds.symmetry = args.symmetry
        ds.n_games = n_games if n_games > 0 else 128
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                            pin_memory=(args.device != "cpu"), drop_last=(len(ds) > args.batch))
        return ds, loader

    # In stream mode, retain recent generations in RAM instead of training
    # once on a generation and immediately discarding it.  This gives the
    # online learner a small replay window while keeping the zero-disk pipe.
    stream_replay_buffer: list[_ReplayDataset] = []

    def prepare_stream_dataset(new_ds: _ReplayDataset):
        stream_replay_buffer.append(new_ds)
        keep = max(1, int(args.replay_keep))
        if len(stream_replay_buffer) > keep:
            del stream_replay_buffer[:-keep]
        merged_ds = merge_weighted_replay_dataset(
            stream_replay_buffer[-1],
            stream_replay_buffer,
            args.symmetry,
        )
        merged_loader = DataLoader(
            merged_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=(args.device != "cpu"),
            drop_last=(len(merged_ds) > args.batch),
        )
        return merged_ds, merged_loader

    # ---- Dataset Setup (Consume-on-Train Queue / RAM Stream) ----------------
    sp_proc = None
    if args.data == "-" or args.data_dir == "-":
        import subprocess
        onnx_model_path = Path(args.out).with_suffix(".onnx").resolve()
        selfplay_games = int(getattr(CONFIG.mcts, "games", 256))
        selfplay_sims = int(getattr(CONFIG.mcts, "sims", 400))
        selfplay_batch = int(getattr(CONFIG.mcts, "gpu_batch_size", 32))
        selfplay_device = str(getattr(CONFIG.device, "rust_device", "gpu"))
        cargo_cmd = [
            "cargo", "run", "--release", "--features", "cuda",
            "--manifest-path", "../src_rust/Cargo.toml", "--",
            "-g", str(selfplay_games),
            "-s", str(selfplay_sims),
            "-b", str(selfplay_batch),
            "-d", selfplay_device,
            "-m", str(onnx_model_path), "-o", "-", "--stream"
        ]
        print("[train] Starting Rust MCTS stream process (ZERO disk I/O, pure RAM IPC)...")
        sp_proc = subprocess.Popen(cargo_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
        ds, loader = None, None
        while ds is None:
            if sp_proc.poll() is not None:
                raise RuntimeError(f"Rust MCTS process exited unexpectedly with code {sp_proc.returncode}")
            ds, loader = read_stream_batch(sp_proc.stdout)
            if ds is not None:
                ds, loader = prepare_stream_dataset(ds)
            if ds is None:
                time.sleep(0.1)
    else:
        # Disk-based dataset loading
        def load_dynamic_dataset(data_dir, data_file, max_samples, symmetry, num_workers_cfg, consume=False):
            if data_dir:
                from dataset import decode_bitboard_batched
                import os, glob
                bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
                if not bin_files:
                    return None, None, []
                
                all_own, all_opp, all_policy, all_value, all_moves_left = [], [], [], [], []
                loaded_files = []
                for f in bin_files:
                    try:
                        sub = C4Dataset(f, max_samples=max_samples)
                        if len(sub) > 0:
                            all_own.append(sub._own)
                            all_opp.append(sub._opp)
                            all_policy.append(sub._policy)
                            all_value.append(sub._value)
                            all_moves_left.append(getattr(sub, "_moves_left", np.zeros_like(sub._value)))
                            loaded_files.append(f)
                    except Exception as e:
                        continue

                if not loaded_files:
                    return None, None, []

                import numpy as np
                own_arr = np.concatenate(all_own)
                opp_arr = np.concatenate(all_opp)
                policy_arr = np.concatenate(all_policy)
                value_arr = np.concatenate(all_value)
                moves_left_arr = np.concatenate(all_moves_left)
                if max_samples is not None and len(own_arr) > max_samples:
                    own_arr, opp_arr = own_arr[:max_samples], opp_arr[:max_samples]
                    policy_arr, value_arr, moves_left_arr = policy_arr[:max_samples], value_arr[:max_samples], moves_left_arr[:max_samples]
                planes_arr = decode_bitboard_batched(own_arr, opp_arr)
                ds = _ReplayDataset(planes_arr, policy_arr, value_arr, moves_left_arr)
                ds.symmetry = symmetry

                if consume:
                    for f in loaded_files:
                        try:
                            os.remove(f)
                        except OSError:
                            pass

                workers = 0
                loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=workers,
                                    pin_memory=(args.device != "cpu"), drop_last=(len(ds) > args.batch))
                return ds, loader, loaded_files
            else:
                ds = C4Dataset(data_file, max_samples=max_samples)
                ds.symmetry = symmetry
                workers = num_workers_cfg if num_workers_cfg is not None else _DEFAULT_NUM_WORKERS
                loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=workers,
                                    pin_memory=(args.device != "cpu"), drop_last=(len(ds) > args.batch))
                return ds, loader, [data_file]

        print(f"[train] Waiting for self-play data from Rust generator in {args.data_dir or args.data}...")
        ds, loader, loaded_files = None, None, []
        while ds is None or len(ds) == 0:
            ds, loader, loaded_files = load_dynamic_dataset(args.data_dir, args.data, args.max_samples, args.symmetry, args.num_workers, consume=True)
            if ds is None or len(ds) == 0:
                time.sleep(0.2)

    # ---- Optimizer / Scheduler / EMA / Compile -----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=args.fused_adamw
    )
    # Replay is sampled into an epoch-sized dataset, so the scheduler must be
    # based on the actual loader length, not on the size of the RAM window.
    estimated_steps_per_epoch = max(1, len(loader))
    schedule_epochs = max(
        int(args.lr_schedule_epochs),
        int(start_epoch + args.epochs),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=schedule_epochs * estimated_steps_per_epoch,
        warmup_steps=args.lr_warmup_epochs * estimated_steps_per_epoch,
        eta_min=args.lr_min,
    )

    use_amp = (not args.no_amp) and args.device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)
    ema = ExponentialMovingAverage(model, decay=args.ema_decay) if args.use_ema else None
    if args.use_ema:
        print(f"[train] Exponential Moving Average (EMA) enabled (decay={args.ema_decay})")

    resume_best_loss = float("inf")
    if resume_training_state is not None:
        try:
            scheduler_state = resume_training_state.get("scheduler_state_dict") or {}
            is_new_scheduler = scheduler_state.get("scheduler_type") == WarmupCosineScheduler.scheduler_type
            if is_new_scheduler:
                scheduler.load_state_dict(scheduler_state)
            optimizer.load_state_dict(resume_training_state["optimizer_state_dict"])
            if not is_new_scheduler:
                # Checkpoints created by the old cosine scheduler are still
                # valid model/Adam checkpoints.  Continue their current LR
                # through a short ramp instead of resetting to args.lr.
                current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
                recovery_peak = [
                    max(current_lr, min(float(args.lr), float(args.lr) * 0.5))
                    for current_lr in current_lrs
                ]
                scheduler.reconfigure_resume(
                    current_lrs,
                    recovery_peak,
                    total_steps=schedule_epochs * estimated_steps_per_epoch,
                    warmup_steps=max(1, args.lr_warmup_epochs * estimated_steps_per_epoch),
                )
                print(
                    "[train] Adapted legacy cosine schedule with a gradual LR recovery "
                    f"({current_lrs[0]:.2e} -> {recovery_peak[0]:.2e})."
                )
            if ema is not None and resume_training_state.get("ema_shadow") is not None:
                ema.shadow = {
                    name: value.to(args.device)
                    for name, value in resume_training_state["ema_shadow"].items()
                }
            scaler_state = resume_training_state.get("scaler_state_dict")
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            resume_best_loss = float(resume_training_state.get("best_loss", float("inf")))
            print(
                "[train] Restored optimizer, scheduler, EMA/scaler state "
                f"(best_loss={resume_best_loss:.4f})."
            )
        except Exception as e:
            print(
                f"[train] WARNING: Could not restore full training state ({e}); "
                "continuing with fresh optimizer state."
            )

    if not args.no_compile and hasattr(torch, "compile"):
        try:
            import triton  # noqa: F401
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[train] torch.compile enabled (mode={args.compile_mode})")
        except ImportError:
            print("[train] triton not installed — skipping torch.compile (use --no-compile to silence this check)")
        except Exception as e:
            print(f"[train] torch.compile failed: {e}; continuing uncompiled")

    # Keep EMA parameter names aligned with the uncompiled model.  Compiled
    # wrappers may expose them as ``_orig_mod.*`` instead.
    inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # ---- Warmup -----------------------------------------------------------
    if not args.no_compile and args.compile_mode != "none":
        print("[train] warming up model to trigger compilation...")
        model.train()
        warmup_state = copy.deepcopy(inner_model.state_dict())
        
        warmup_steps = 10
        steps_done = 0
        if len(ds) > 0:
            while steps_done < warmup_steps:
                for warmup_batch in loader:
                    wp, wtp, wtv, wtm = [x.to(args.device) for x in warmup_batch]
                    if args.channels_last:
                        wp = wp.to(memory_format=torch.channels_last)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=use_amp, device_type="cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else None):
                        w_log_p, w_v, w_m, _w_c = model(wp)
                        w_loss, _, _, _ = compute_loss(
                            w_log_p,
                            w_v,
                            w_m,
                            wtp,
                            wtv,
                            wtm,
                            _w_c,
                            args.confidence_loss_weight,
                        )
                    scaler.scale(w_loss).backward()
                    # Compile with a real backward pass, but do not consume
                    # optimizer/scheduler steps or alter resumed weights.
                    
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    
                    steps_done += 1
                    if steps_done >= warmup_steps:
                        break
        inner_model.load_state_dict(warmup_state)
        optimizer.zero_grad(set_to_none=True)
        print("[train] warmup complete.")

    # Build the IPC graph once before the timed training loop.  Subsequent
    # epochs only replace its initializers, avoiding a full ONNX export during
    # epoch 1 (which is especially expensive with torch.export/ONNXScript).
    ipc_onnx_template: bytes | None = None
    if sp_proc is not None and not args.no_onnx:
        try:
            ipc_inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            print("[train] preparing reusable ONNX IPC template before epoch 1...")
            ipc_onnx_template = export_onnx_to_bytes(
                ipc_inner,
                opset=args.onnx_opset,
                infer_precision=args.infer_precision,
            )
            print(f"[train] ONNX IPC template ready ({len(ipc_onnx_template):,} bytes).")
        except Exception as e:
            print(f"[train] WARNING: Could not prepare ONNX IPC template: {e}")

    # ---- Train Loop -------------------------------------------------------
    best_loss = resume_best_loss
    global_start_t = time.time()
    total_samples = 0
    epoch = 0
    done = False
    _prev = None
    ema_model = None

    import signal
    def handle_sigint(sig, frame):
        nonlocal done
        print("\n[train] Caught KeyboardInterrupt (Ctrl+C). Gracefully stopping and saving the model...")
        done = True
    signal.signal(signal.SIGINT, handle_sigint)

    while not done:
        t_epoch_start = time.time()
        model.train()

        # 1. Measure dataset reload / queue waiting time
        t_data_start = time.time()
        if sp_proc is not None:
            # Direct RAM streaming from Rust stdout pipe
            if sp_proc.poll() is not None:
                if done: break
                raise RuntimeError(f"Rust MCTS process exited unexpectedly with code {sp_proc.returncode}")
            ds, loader = read_stream_batch(sp_proc.stdout)
            if ds is not None:
                ds, loader = prepare_stream_dataset(ds)
            while ds is None:
                if sp_proc.poll() is not None:
                    if done: break
                    raise RuntimeError(f"Rust MCTS process exited unexpectedly with code {sp_proc.returncode}")
                if done: break
                time.sleep(0.1)
                ds, loader = read_stream_batch(sp_proc.stdout)
                if ds is not None:
                    ds, loader = prepare_stream_dataset(ds)
            if done: break
        elif args.data_dir and epoch > 0:
            # Poll data_dir until fresh self-play data arrives from Rust generator
            new_ds, new_loader, _ = load_dynamic_dataset(args.data_dir, args.data, args.max_samples, args.symmetry, args.num_workers, consume=True)
            while new_ds is None or len(new_ds) == 0:
                time.sleep(0.1)
                new_ds, new_loader, _ = load_dynamic_dataset(args.data_dir, args.data, args.max_samples, args.symmetry, args.num_workers, consume=True)
            ds, loader = new_ds, new_loader
        t_data = time.time() - t_data_start

        if _prev is not None:
            (p_se, p_ee, p_ns, p_avg_tot, p_avg_pol, p_avg_val, p_avg_mvl, p_lr, p_v, p_estr, p_bl, p_tdata, p_texport, p_ttot, p_tfwd, p_tbwd, p_topt) = _prev
            if args.device == "cuda":
                gpu_ms = p_se.elapsed_time(p_ee)
                p_train_sec = gpu_ms / 1000.0
            else:
                p_train_sec = p_ee
            p_sps = p_ns / max(0.001, p_train_sec)
            p_tother = max(0.0, p_ttot - p_train_sec - p_tdata - p_texport)
            total_time_so_far = time.time() - global_start_t
            n_g = getattr(ds, "n_games", 0)
            print(
                f"[train] {p_estr} done in {format_duration(p_ttot)} "
                f"(train={format_duration(p_train_sec)} "
                f"(bwd={format_duration(p_tbwd)}, fwd={format_duration(p_tfwd)}, opt={format_duration(p_topt)}), "
                f"data={format_duration(p_tdata)}, ema/export={format_duration(p_texport)}, "
                f"other={format_duration(p_tother)} | tot={format_duration(total_time_so_far)}) "
                f"({p_sps:.0f} samples/s) | loss={p_avg_tot:.4f} "
                f"(policy={p_avg_pol:.4f}, value={p_avg_val:.4f}, mvl={p_avg_mvl:.4f}) lr={p_lr:.2e} | "
                f"samples={len(ds):,} | games={n_g:,}"
            )
            if p_bl and p_avg_tot < best_loss:
                best_loss = p_avg_tot

        # 2. Measure CUDA training time
        if args.device == "cuda":
            epoch_start_evt = torch.cuda.Event(enable_timing=True)
            epoch_end_evt   = torch.cuda.Event(enable_timing=True)
            epoch_start_evt.record()
        else:
            epoch_start_evt = time.time()

        t0 = time.time()
        running = {"total": 0.0, "policy": 0.0, "value": 0.0, "moves_left": 0.0}
        log_running = {"total": 0.0, "policy": 0.0, "value": 0.0, "moves_left": 0.0}
        n_batches = 0
        log_batches = 0
        samples_this_epoch = 0
        fwd_sec_epoch = 0.0
        bwd_sec_epoch = 0.0
        opt_sec_epoch = 0.0
        prefetcher = CudaPrefetcher(loader, torch.device(args.device), channels_last=args.channels_last)

        batch_idx = 0
        calibration_batch = None
        while True:
            batch = prefetcher.next()
            if batch is None: break
            planes, target_policy, target_value, target_moves_left = batch
            
            if calibration_batch is None and planes.size(0) > 0:
                # Save a small subset of the first batch for INT8 static calibration
                # Convert to float32 numpy arrays as expected by ONNX
                calibration_batch = planes[:128].detach().cpu().to(torch.float32).numpy()
            
            if args.duration is not None and time.time() - global_start_t >= args.duration:
                done = True
                break

            optimizer.zero_grad(set_to_none=True)

            t_f0 = time.perf_counter()
            with autocast(enabled=use_amp, device_type="cuda" if use_amp else "cpu", dtype=torch.float16 if use_amp else None):
                log_p, v, m, _c = model(planes)
                loss, policy_loss, value_loss, moves_left_loss = compute_loss(
                    log_p,
                    v,
                    m,
                    target_policy,
                    target_value,
                    target_moves_left,
                    _c,
                    args.confidence_loss_weight,
                )
            if args.device == "cuda": torch.cuda.synchronize()
            fwd_sec_epoch += (time.perf_counter() - t_f0)

            t_b0 = time.perf_counter()
            scaler.scale(loss).backward()
            if args.device == "cuda": torch.cuda.synchronize()
            bwd_sec_epoch += (time.perf_counter() - t_b0)

            t_o0 = time.perf_counter()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_DEFAULT_MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                # Update against the uncompiled module so parameter names
                # match the EMA shadow keys (compiled wrappers may add
                # ``_orig_mod.`` to names).
                ema.update(inner_model)
            scheduler.step()
            if args.device == "cuda": torch.cuda.synchronize()
            opt_sec_epoch += (time.perf_counter() - t_o0)

            running["total"] += loss.detach()
            running["policy"] += policy_loss.detach()
            running["value"] += value_loss.detach()
            running["moves_left"] += moves_left_loss.detach()
            log_running["total"] += loss.detach()
            log_running["policy"] += policy_loss.detach()
            log_running["value"] += value_loss.detach()
            log_running["moves_left"] += moves_left_loss.detach()
            n_batches += 1
            log_batches += 1
            total_samples += planes.size(0)
            samples_this_epoch += planes.size(0)

            if args.log_every > 0 and (batch_idx + 1) % args.log_every == 0:
                avg = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v) / max(1, log_batches)
                    for k, v in log_running.items()
                }
                lr = scheduler.get_last_lr()[0]
                global_epoch = start_epoch + epoch + 1
                epoch_str = (
                    f"epoch {global_epoch}/{start_epoch + args.epochs}"
                    if args.duration is None
                    else f"epoch {global_epoch}"
                )
                print(
                    f"[train] {epoch_str}  "
                    f"batch {batch_idx+1}/{len(loader)}  "
                    f"loss={avg['total']:.4f}  policy={avg['policy']:.4f}  "
                    f"value={avg['value']:.4f}  mvl={avg['moves_left']:.4f}  lr={lr:.2e}  "
                    f"({format_duration(time.time() - t0)})"
                )
                log_running = {"total": 0.0, "policy": 0.0, "value": 0.0, "moves_left": 0.0}
                log_batches = 0
            batch_idx += 1

        if batch_idx == 0:
            break

        if args.device == "cuda":
            epoch_end_evt.record()
        else:
            epoch_end_evt = time.time() - epoch_start_evt

        # 3. Measure model save & ONNX export time (both raw & EMA)
        t_export_start = time.time()
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        
        # Save PyTorch models (.pt) to disk only when loss improves or training completes
        is_best_epoch = (running["total"].item() / max(1, n_batches)) < best_loss
        if is_best_epoch or (epoch == args.epochs - 1) or done:
            inner.save(args.out)
            if 'ema' in locals() and ema is not None:
                ema_model = copy.deepcopy(inner)
                ema.apply_shadow(ema_model)
                ema_pt_path = args.out.replace(".pt", "_ema.pt")
                ema_model.save(ema_pt_path)
            else:
                ema_model = None
        else:
            ema_model = None

        should_export_onnx = (not args.no_onnx) and (
            (args.onnx_every > 0 and (epoch + 1) % args.onnx_every == 0)
            or (epoch == args.epochs - 1)
            or done
        )
        if should_export_onnx:
            if sp_proc is not None and sp_proc.stdin is not None:
                try:
                    # Export directly to a self-contained in-memory ONNX
                    # blob.  Path-based exports may create a companion
                    # `.onnx.data` file, which cannot be transferred through
                    # this one-buffer stdin protocol.
                    data = None
                    if ipc_onnx_template is not None and args.infer_precision == "fp32":
                        data = update_onnx_weights_bytes(inner, ipc_onnx_template)
                    if data is None:
                        data = export_onnx_to_bytes(
                            inner,
                            opset=args.onnx_opset,
                            infer_precision=args.infer_precision,
                        )
                    ipc_onnx_template = data
                    sp_proc.stdin.write(len(data).to_bytes(4, byteorder="little"))
                    sp_proc.stdin.write(data)
                    sp_proc.stdin.flush()
                except Exception as e:
                    print(f"[train] WARNING: ONNX IPC export to Rust failed: {e}")
            else:
                # Fallback to disk if not in stream mode
                onnx_path = args.out.replace(".pt", ".onnx")
                try:
                    export_onnx(inner, onnx_path, opset=args.onnx_opset, infer_precision=args.infer_precision, calibration_samples=calibration_batch)
                except Exception as e:
                    pass
                
                # Export EMA ONNX model
                if ema_model is not None:
                    onnx_ema_path = args.out.replace(".pt", "_ema.onnx")
                    try:
                        export_onnx(ema_model, onnx_ema_path, opset=args.onnx_opset, infer_precision=args.infer_precision, calibration_samples=calibration_batch)
                    except Exception as e:
                        pass

        t_export = time.time() - t_export_start
        t_epoch_total = time.time() - t_epoch_start

        v_item = v[-1].item() if isinstance(v, torch.Tensor) and v.numel() > 0 else 0.0
        avg_tot = running["total"].item() / max(1, n_batches) if isinstance(running["total"], torch.Tensor) else 0.0
        avg_pol = running["policy"].item() / max(1, n_batches) if isinstance(running["policy"], torch.Tensor) else 0.0
        avg_val = running["value"].item() / max(1, n_batches) if isinstance(running["value"], torch.Tensor) else 0.0
        avg_mvl = running["moves_left"].item() / max(1, n_batches) if isinstance(running["moves_left"], torch.Tensor) else 0.0
        lr = scheduler.get_last_lr()[0]

        global_epoch = start_epoch + epoch + 1
        epoch_str = (
            f"epoch {global_epoch:2d}/{start_epoch + args.epochs}"
            if args.duration is None
            else f"epoch {global_epoch:2d}"
        )
        _prev = (epoch_start_evt, epoch_end_evt, samples_this_epoch,
                 avg_tot, avg_pol, avg_val, avg_mvl, lr, v_item, epoch_str, n_batches > 0,
                 t_data, t_export, t_epoch_total, fwd_sec_epoch, bwd_sec_epoch, opt_sec_epoch)

        if done or (args.duration is None and epoch >= args.epochs - 1):
            done = True
        
        epoch += 1

    # Print the last epoch (sync after training is done, no pipeline to preserve)
    if _prev is not None:
        (p_se, p_ee, p_ns, p_avg_tot, p_avg_pol, p_avg_val, p_avg_mvl, p_lr, p_v, p_estr, p_bl, p_tdata, p_texport, p_ttot, p_tfwd, p_tbwd, p_topt) = _prev
        if args.device == "cuda":
            p_se.synchronize()
            gpu_ms = p_se.elapsed_time(p_ee)
            p_train_sec = gpu_ms / 1000.0
        else:
            p_train_sec = p_ee
        p_sps = p_ns / max(0.001, p_train_sec)
        p_tother = max(0.0, p_ttot - p_train_sec - p_tdata - p_texport)
        total_time_so_far = time.time() - global_start_t
        n_g = getattr(ds, "n_games", 0)
        print(
            f"[train] {p_estr} done in {format_duration(p_ttot)} "
            f"(train={format_duration(p_train_sec)} "
            f"(bwd={format_duration(p_tbwd)}, fwd={format_duration(p_tfwd)}, opt={format_duration(p_topt)}), "
            f"data={format_duration(p_tdata)}, ema/export={format_duration(p_texport)}, "
            f"other={format_duration(p_tother)} | tot={format_duration(total_time_so_far)}) "
            f"({p_sps:.0f} samples/s) | loss={p_avg_tot:.4f} "
            f"(policy={p_avg_pol:.4f}, value={p_avg_val:.4f}, mvl={p_avg_mvl:.4f}) lr={p_lr:.2e} | "
            f"samples={len(ds):,} | games={n_g:,} | sample val={p_v:+.3f}"
        )
        if p_bl and p_avg_tot < best_loss:
            best_loss = p_avg_tot


    if args.device == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - global_start_t
    overall_throughput = total_samples / max(0.001, total_time)
    print(f"[train] overall throughput: {overall_throughput:.1f} samples/s")

    # Save the model ONCE at the very end
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner.save(args.out)
    final_epoch = start_epoch + epoch
    print(f"[train] saved final model (loss={best_loss:.4f}) to {args.out}")

    # Keep the pipeline's stable output path, and also create a cumulative
    # checkpoint so a resumed run can continue with the correct epoch number.
    if final_epoch > start_epoch:
        epoch_checkpoint = checkpoint_path_for_epoch(args.out, final_epoch)
        if epoch_checkpoint.resolve() != out_path.resolve():
            save_training_checkpoint(
                epoch_checkpoint,
                inner,
                optimizer,
                scheduler,
                ema,
                scaler,
                best_loss,
            )
            print(f"[train] saved cumulative checkpoint (epoch {final_epoch}) to {epoch_checkpoint}")
            if ema_model is not None:
                ema_checkpoint = epoch_checkpoint.with_name(
                    f"{epoch_checkpoint.stem}_ema{epoch_checkpoint.suffix}"
                )
                ema_model.save(ema_checkpoint)

    # ---- ONNX export (consumed by the Rust MCTS) ------------------------
    if not args.no_onnx:
        onnx_path = args.out.replace(".pt", ".onnx")
        print(f"[train] exporting ONNX -> {onnx_path} (opset {args.onnx_opset}, precision: {args.infer_precision})")
        # Use the un-compiled underlying model.
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        try:
            export_onnx(inner, onnx_path, opset=args.onnx_opset, infer_precision=args.infer_precision)
            print(f"[train] ONNX export OK. Next self-play cycle will use it.")
        except Exception as e:
            print(f"[train] WARNING: ONNX export failed: {e}")
            print(f"[train] Rust MCTS will fall back to null network on the next cycle.")

    if sp_proc is not None:
        try:
            sp_proc.terminate()
        except Exception:
            pass

    print(f"[train] done. best epoch loss = {best_loss:.4f}")


if __name__ == "__main__":
    main()
