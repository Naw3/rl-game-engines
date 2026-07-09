"""
init.py — Bootstrap a random-init Connect4Net and export to ONNX.

This is the cycle-0 bootstrap. Before the very first self-play, we
need *some* network — even a random-weight one — so the MCTS has
priors and value estimates to work with. Without this, the MCTS would
fall back to the null network (uniform priors, value=0), which is
*technically* valid but produces a uselessly uniform dataset on cycle
0. With a random-init model, cycle 0's dataset has at least some
shape — the network's random priors give the PUCT search a non-flat
starting point, and the value head's random output gives the Q values
some variance. After training on that, cycle 1 has a real network.

CLI
---
    python init.py [--out-pt connect4_model.pt]
                   [--out-onnx connect4_model.onnx]
                   [--opset 17]
                   [--force]

If the ONNX file already exists, init.py is a no-op (use --force to
overwrite). This makes it safe to call from the orchestrator every
cycle — it just becomes a no-op after the first time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr. Windows defaults to cp1252 which crashes on
# Unicode arrows in print() output. Safe on all platforms.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from model import Connect4Net


# Resolve the project root from this script's location. We use this as the
# default location for the .pt and .onnx files so they end up in the same
# place the orchestrator and the Rust CLI look for them. Callers can still
# override with --out-pt / --out-onnx.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PT = str(_PROJECT_ROOT / "connect4_model.pt")
_DEFAULT_ONNX = str(_PROJECT_ROOT / "connect4_model.onnx")


def export_onnx(
    model: Connect4Net,
    onnx_path: str,
    opset: int = 18,
) -> None:
    """Export `model` to ONNX with dynamic batch dim and named I/O.

    The exported graph has:
        input  "input"  shape (batch, 3, 6, 7)  f32  (batch is dynamic)
        output "policy" shape (batch, 7)        f32  (log-probabilities)
        output "value"  shape (batch,)          f32  (in [-1, 1] via tanh)

    The Rust side (network.rs) reads "policy" and "value" by name and
    softmaxes the policy (since the model head outputs log-softmax).
    """
    model.eval()
    dummy = torch.randn(1, 3, 6, 7)

    # Use dynamic_shapes (preferred over deprecated dynamic_axes with dynamo=True).
    batch = torch.export.Dim("batch", min=1, max=256)
    dynamic_shapes = {"input": {0: batch}}

    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        do_constant_folding=True,
    )
    # Smoke-check: re-load and run.
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap random-init Connect4Net + ONNX export")
    p.add_argument("--out-pt", default=_DEFAULT_PT,
                   help=f"PyTorch state_dict output path (default {_DEFAULT_PT})")
    p.add_argument("--out-onnx", default=_DEFAULT_ONNX,
                   help=f"ONNX output path, consumed by Rust MCTS (default {_DEFAULT_ONNX})")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=3)
    args = p.parse_args()

    if os.path.exists(args.out_onnx) and not args.force:
        print(f"[init] {args.out_onnx} already exists — nothing to do (use --force to overwrite)")
        return 0

    print(f"[init] creating random-init Connect4Net "
          f"(channels={args.channels}, num_blocks={args.num_blocks})")
    net = Connect4Net(channels=args.channels, num_blocks=args.num_blocks)
    print(f"[init] model: {net.num_parameters():,} parameters")

    print(f"[init] saving state_dict -> {args.out_pt}")
    net.save(args.out_pt)

    print(f"[init] exporting ONNX -> {args.out_onnx} (opset {args.opset})")
    export_onnx(net, args.out_onnx, opset=args.opset)

    print(f"[init] done. {args.out_pt} + {args.out_onnx} ready for self-play.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
