"""
init.py — Bootstrap a random-init Connect4Net + ONNX export.

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
                   [--opset 18]
                   [--force]

If the ONNX file already exists, init.py is a no-op (use --force to
overwrite). This makes it safe to call from the orchestrator every
cycle — it just becomes a no-op after the first time.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr. Windows defaults to cp1252 which crashes on
# Unicode arrows in print() output. Safe on all platforms.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings
import logging

warnings.filterwarnings("ignore")
for _log_name in ["torch.onnx", "torch.onnx._internal", "torch.export"]:
    logging.getLogger(_log_name).setLevel(logging.ERROR)

import torch
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model import Connect4Net
from utils.onnx_export import export_onnx

try:
    from config import CONFIG
    _DEFAULT_PT              = str(CONFIG.paths.model_pt)
    _DEFAULT_ONNX            = str(CONFIG.paths.model_onnx)
    _DEFAULT_CHANNELS        = CONFIG.network.channels
    _DEFAULT_NUM_BLOCKS      = CONFIG.network.num_blocks
    _DEFAULT_SEED            = CONFIG.mcts.seed
    _DEFAULT_OPSET           = CONFIG.dataset.onnx_opset
    _DEFAULT_PLANES          = CONFIG.network.input_planes
    _DEFAULT_ROWS            = CONFIG.network.board_rows
    _DEFAULT_COLS            = CONFIG.network.board_cols
    _DEFAULT_MAX_ONNX_BATCH  = CONFIG.dataset.max_onnx_batch
    _DEFAULT_INFER_PRECISION = CONFIG.train.infer_precision
except Exception as err:
    print(f"[init] WARNING: Failed to load config.py ({err}); using fallbacks")
    _DEFAULT_PT              = str(_PROJECT_ROOT / "connect4_model.pt")
    _DEFAULT_ONNX            = str(_PROJECT_ROOT / "connect4_model.onnx")
    _DEFAULT_CHANNELS        = 64
    _DEFAULT_NUM_BLOCKS      = 3
    _DEFAULT_SEED            = 42
    _DEFAULT_OPSET           = 18
    _DEFAULT_PLANES          = 3
    _DEFAULT_ROWS            = 6
    _DEFAULT_COLS            = 7
    _DEFAULT_MAX_ONNX_BATCH  = 256
    _DEFAULT_INFER_PRECISION = "fp32"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Bootstrap random-init Connect4Net + ONNX export"
    )
    p.add_argument(
        "--out-pt", default=_DEFAULT_PT,
        help=f"PyTorch state_dict output path (default {_DEFAULT_PT})",
    )
    p.add_argument(
        "--out-onnx", default=_DEFAULT_ONNX,
        help=f"ONNX output path, consumed by Rust MCTS (default {_DEFAULT_ONNX})",
    )
    p.add_argument("--opset", type=int, default=_DEFAULT_OPSET,
                   help="ONNX opset version")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files")
    p.add_argument("--channels", type=int, default=_DEFAULT_CHANNELS)
    p.add_argument("--num-blocks", type=int, default=_DEFAULT_NUM_BLOCKS)
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED,
                   help="RNG seed for model initialization")
    p.add_argument(
        "--infer-precision", choices=["fp32", "fp16", "int8"],
        default=_DEFAULT_INFER_PRECISION,
        help="Precision for the exported ONNX model",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if os.path.exists(args.out_onnx) and not args.force:
        print(
            f"[init] {args.out_onnx} already exists — nothing to do "
            "(use --force to overwrite)"
        )
        return 0

    print(
        f"[init] creating random-init Connect4Net "
        f"(channels={args.channels}, num_blocks={args.num_blocks})"
    )
    net = Connect4Net(channels=args.channels, num_blocks=args.num_blocks)
    print(f"[init] model: {net.num_parameters():,} parameters")

    print(f"[init] saving state_dict -> {args.out_pt}")
    net.save(args.out_pt)

    print(
        f"[init] exporting ONNX -> {args.out_onnx} "
        f"(opset {args.opset}, precision: {args.infer_precision})"
    )
    export_onnx(net, args.out_onnx, opset=args.opset,
                infer_precision=args.infer_precision)

    print(f"[init] done. {args.out_pt} + {args.out_onnx} ready for self-play.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
