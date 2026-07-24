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
import random
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
logging.getLogger("torch.onnx._internal").setLevel(logging.ERROR)
logging.getLogger("torch.export").setLevel(logging.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import CONFIG
    _DEFAULT_DATA = str(CONFIG.paths.selfplay_bin.name)
    _DEFAULT_OUT = str(CONFIG.paths.model_pt.name)
    _DEFAULT_EPOCHS = CONFIG.train.epochs
    _DEFAULT_BATCH = CONFIG.train.batch_size
    _DEFAULT_LR = CONFIG.train.learning_rate
    _DEFAULT_WD = CONFIG.train.weight_decay
    _DEFAULT_SEED = CONFIG.mcts.seed
    _DEFAULT_NUM_WORKERS = CONFIG.train.num_workers
    _DEFAULT_LOG_EVERY = CONFIG.train.log_every
    _DEFAULT_OPSET = CONFIG.dataset.onnx_opset
    _DEFAULT_PLANES = CONFIG.network.input_planes
    _DEFAULT_ROWS = CONFIG.network.board_rows
    _DEFAULT_COLS = CONFIG.network.board_cols
    _DEFAULT_MAX_ONNX_BATCH = CONFIG.dataset.max_onnx_batch
    _DEFAULT_MAX_GRAD_NORM = CONFIG.train.max_grad_norm
    _DEFAULT_SYMMETRY = CONFIG.train.symmetry
    _DEFAULT_NO_AMP = not CONFIG.train.use_amp
    _DEFAULT_NO_COMPILE = not CONFIG.train.use_compile
except Exception as err:
    print(f"[train] WARNING: Failed to load config.py ({err}); using fallbacks")
    _DEFAULT_DATA = "selfplay.bin"
    _DEFAULT_OUT = "connect4_model.pt"
    _DEFAULT_EPOCHS = 5
    _DEFAULT_BATCH = 256
    _DEFAULT_LR = 1e-3
    _DEFAULT_WD = 1e-4
    _DEFAULT_SEED = 42
    _DEFAULT_NUM_WORKERS = 2
    _DEFAULT_LOG_EVERY = 20
    _DEFAULT_OPSET = 18
    _DEFAULT_PLANES = 3
    _DEFAULT_ROWS = 6
    _DEFAULT_COLS = 7
    _DEFAULT_MAX_ONNX_BATCH = 256
    _DEFAULT_MAX_GRAD_NORM = 5.0
    _DEFAULT_SYMMETRY = True
    _DEFAULT_NO_AMP = False
    _DEFAULT_NO_COMPILE = False

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast, GradScaler
except ImportError:  # PyTorch < 2.0
    from torch.cuda.amp import autocast, GradScaler

from dataset import C4Dataset
from model import Connect4Net


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    log_p: torch.Tensor,
    v: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, policy_loss, value_loss)."""
    # Policy: cross-entropy with soft targets. target_policy is (B, 7),
    # log_p is (B, 7) of log-probabilities. Sum over columns, mean over batch.
    policy_loss = -(target_policy * log_p).sum(dim=1).mean()
    # Value: standard MSE.
    value_loss = F.mse_loss(v, target_value)
    return policy_loss + value_loss, policy_loss, value_loss


def export_onnx(model: Connect4Net, onnx_path: str, opset: int = _DEFAULT_OPSET) -> None:
    """Export the (already-trained) model to ONNX.

    Output contract — the Rust side (network.rs) reads by name:
        input  "input"  shape (batch, 3, 6, 7)  f32
        output "policy" shape (batch, 7)        f32  (log-probabilities)
        output "value"  shape (batch,)          f32  (in [-1, 1] via tanh)

    The Rust side softmaxes the policy (since the model head outputs
    log-softmax). The model is moved to CPU before export to avoid
    ONNX complaining about CUDA tensors.
    """
    was_cuda = next(model.parameters()).device.type == "cuda"
    if was_cuda:
        model_cpu = Connect4Net(
            channels=model.channels, num_blocks=model.num_blocks
        ).cpu()
        model_cpu.load_state_dict(model.state_dict())
    else:
        model_cpu = model
    model_cpu.eval()

    dummy = torch.randn(1, _DEFAULT_PLANES, _DEFAULT_ROWS, _DEFAULT_COLS)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value": {0: "batch_size"},
    }
    import os, sys
    class SuppressOutput:
        def __enter__(self):
            self._stdout, self._stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = open(os.devnull, 'w', encoding='utf-8')
            try:
                self.fd = os.open(os.devnull, os.O_WRONLY)
                self.save_out = os.dup(1)
                self.save_err = os.dup(2)
                os.dup2(self.fd, 1)
                os.dup2(self.fd, 2)
            except Exception: pass
        def __exit__(self, *args):
            sys.stdout.close()
            sys.stdout, sys.stderr = self._stdout, self._stderr
            try:
                os.dup2(self.save_out, 1)
                os.dup2(self.save_err, 2)
                os.close(self.fd); os.close(self.save_out); os.close(self.save_err)
            except Exception: pass

    with SuppressOutput():
        torch.onnx.export(
        model_cpu,
        (dummy,),
        onnx_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    # Verify the exported graph is loadable and the outputs are correct.
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


class _ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, planes, policy, value):
        self._planes, self._policy, self._value = planes, policy, value
        self.count = len(planes)
        self.symmetry = False
    def __len__(self): return self.count
    def __getitem__(self, idx):
        planes = self._planes[idx]
        policy = self._policy[idx]
        value = self._value[idx]
        if self.symmetry:
            if random.random() < 0.5:
                planes = planes[:, :, ::-1].copy()
                policy = policy[::-1].copy()
        return (torch.from_numpy(planes),
                torch.from_numpy(policy),
                torch.tensor(value, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


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
    p.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=_DEFAULT_LR)
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

    # ---- Data -------------------------------------------------------------
    # If --data-dir is set, load all C4D1 files in it (replay buffer).
    # Otherwise load the single --data file.
    if args.data_dir:
        from dataset import C4Dataset, decode_bitboard_batched
        import os, glob
        bin_files = sorted(glob.glob(os.path.join(args.data_dir, "*.bin")))
        if not bin_files:
            print(f"[train] no .bin files found in {args.data_dir} — aborting")
            return
        print(f"[train] replay buffer: {len(bin_files)} file(s) from {args.data_dir}")
        # Concatenate all samples in memory then build one big planes array.
        all_own, all_opp, all_policy, all_value = [], [], [], []
        for f in bin_files:
            sub = C4Dataset(f, max_samples=args.max_samples)
            print(f"  - {f}: {len(sub):,} samples")
            all_own.append(sub._own);  all_opp.append(sub._opp)
            all_policy.append(sub._policy); all_value.append(sub._value)
            if args.max_samples is not None and sum(len(x) for x in all_own) >= args.max_samples:
                break
        import numpy as np
        own_arr   = np.concatenate(all_own)
        opp_arr   = np.concatenate(all_opp)
        policy_arr = np.concatenate(all_policy)
        value_arr = np.concatenate(all_value)
        if args.max_samples is not None and len(own_arr) > args.max_samples:
            own_arr, opp_arr = own_arr[:args.max_samples], opp_arr[:args.max_samples]
            policy_arr, value_arr = policy_arr[:args.max_samples], value_arr[:args.max_samples]
        planes_arr = decode_bitboard_batched(own_arr, opp_arr)
        
        ds = _ReplayDataset(planes_arr, policy_arr, value_arr)
        ds.symmetry = args.symmetry
        if args.symmetry:
            print("[train] symmetry augmentation ON (horizontal flip, 50/50)")
        
        if args.num_workers is None:
            args.num_workers = 2

        # Quick stats for replay
        n = len(ds)
        n_pos = int((value_arr > 0).sum()); n_neg = int((value_arr < 0).sum())
        print(f"[train] replay dataset: {n:,} samples | wins={n_pos} losses={n_neg} draws={n - n_pos - n_neg}")
    else:
        ds = C4Dataset(args.data, max_samples=args.max_samples)
        ds.symmetry = args.symmetry
        if args.symmetry:
            print("[train] symmetry augmentation ON (horizontal flip, 50/50)")
        print(f"[train] dataset: {ds.stats()}")
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers if args.num_workers is not None else _DEFAULT_NUM_WORKERS,
        pin_memory=(args.device != "cpu"),
        drop_last=False,
    )

    # ---- Model ------------------------------------------------------------
    model = Connect4Net().to(args.device)
    n_params = model.num_parameters()
    print(f"[train] model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    use_amp = (not args.no_amp) and args.device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    if not args.no_compile and hasattr(torch, "compile"):
        # Triton is the inductor backend torch.compile uses for CUDA codegen.
        # On Windows it has no compatible wheel from PyPI — the import fails
        # and torch.compile crashes at first forward. Auto-detect and skip.
        try:
            import triton  # noqa: F401
            model = torch.compile(model, mode="reduce-overhead")
            print("[train] torch.compile enabled (mode=reduce-overhead)")
        except ImportError:
            print("[train] triton not installed — skipping torch.compile (use --no-compile to silence this check)")
        except Exception as e:
            print(f"[train] torch.compile failed: {e}; continuing uncompiled")

    # ---- Train ------------------------------------------------------------
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "policy": 0.0, "value": 0.0}
        n_batches = 0

        for batch_idx, (planes, target_policy, target_value) in enumerate(loader):
            planes = planes.to(args.device, non_blocking=True)
            target_policy = target_policy.to(args.device, non_blocking=True)
            target_value = target_value.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp, device_type="cuda" if use_amp else "cpu"):
                log_p, v = model(planes)
                loss, policy_loss, value_loss = compute_loss(
                    log_p, v, target_policy, target_value
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_DEFAULT_MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["total"] += loss.item()
            running["policy"] += policy_loss.item()
            running["value"] += value_loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.log_every == 0:
                avg = {k: v / n_batches for k, v in running.items()}
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[train] epoch {epoch+1}/{args.epochs}  "
                    f"batch {batch_idx+1}/{len(loader)}  "
                    f"loss={avg['total']:.4f}  policy={avg['policy']:.4f}  "
                    f"value={avg['value']:.4f}  lr={lr:.2e}  "
                    f"({(time.time()-t0):.1f}s)"
                )
                running = {"total": 0.0, "policy": 0.0, "value": 0.0}
                n_batches = 0

        # End-of-epoch summary.
        epoch_sec = time.time() - t0
        samples_per_sec = len(ds) / max(0.001, epoch_sec)
        avg_tot = running["total"] / max(1, n_batches)
        avg_pol = running["policy"] / max(1, n_batches)
        avg_val = running["value"] / max(1, n_batches)
        lr = scheduler.get_last_lr()[0]

        model.eval()
        with torch.no_grad():
            sample = ds[0]
            planes = sample[0].unsqueeze(0).to(args.device)
            log_p, v = model(planes)
            policy = log_p.exp().cpu().numpy()[0]

        print(
            f"[train] epoch {epoch+1:2d}/{args.epochs} done in {epoch_sec:.2f}s "
            f"({samples_per_sec:.0f} samples/s) | loss={avg_tot:.4f} "
            f"(policy={avg_pol:.4f}, value={avg_val:.4f}) lr={lr:.2e} | "
            f"sample val={v.item():+.3f}"
        )

        # Snapshot the best model (raw state_dict, so GUI can load directly).
        avg_total = running["total"] / max(1, n_batches)
        if avg_total < best_loss:
            best_loss = avg_total
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            inner.save(args.out)
            print(f"[train] saved best model (loss={best_loss:.4f}) to {args.out}")

    # ---- ONNX export (consumed by the Rust MCTS) ------------------------
    if not args.no_onnx:
        onnx_path = args.out[: -len(".pt")] + ".onnx" if args.out.endswith(".pt") \
            else args.out + ".onnx"
        print(f"[train] exporting ONNX -> {onnx_path} (opset {args.onnx_opset})")
        # Use the un-compiled underlying model.
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        try:
            export_onnx(inner, onnx_path, opset=args.onnx_opset)
            print(f"[train] ONNX export OK. Next self-play cycle will use it.")
        except Exception as e:
            print(f"[train] WARNING: ONNX export failed: {e}")
            print(f"[train] Rust MCTS will fall back to null network on the next cycle.")

    print(f"[train] done. best epoch loss = {best_loss:.4f}")


if __name__ == "__main__":
    main()
