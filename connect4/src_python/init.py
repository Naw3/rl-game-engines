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

import warnings
warnings.filterwarnings("ignore")
import logging
for _log_name in ["torch.onnx", "torch.onnx._internal", "torch.export"]:
    logging.getLogger(_log_name).setLevel(logging.ERROR)

import torch

from model import Connect4Net


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import CONFIG
    _DEFAULT_PT = str(CONFIG.paths.model_pt)
    _DEFAULT_ONNX = str(CONFIG.paths.model_onnx)
    _DEFAULT_CHANNELS = CONFIG.network.channels
    _DEFAULT_NUM_BLOCKS = CONFIG.network.num_blocks
    _DEFAULT_SEED = CONFIG.mcts.seed
    _DEFAULT_OPSET = CONFIG.dataset.onnx_opset
    _DEFAULT_PLANES = CONFIG.network.input_planes
    _DEFAULT_ROWS = CONFIG.network.board_rows
    _DEFAULT_COLS = CONFIG.network.board_cols
    _DEFAULT_MAX_ONNX_BATCH = CONFIG.dataset.max_onnx_batch
except Exception as err:
    print(f"[init] WARNING: Failed to load config.py ({err}); using fallbacks")
    _DEFAULT_PT = str(_PROJECT_ROOT / "connect4_model.pt")
    _DEFAULT_ONNX = str(_PROJECT_ROOT / "connect4_model.onnx")
    _DEFAULT_CHANNELS = 64
    _DEFAULT_NUM_BLOCKS = 3
    _DEFAULT_SEED = 42
    _DEFAULT_OPSET = 18
    _DEFAULT_PLANES = 3
    _DEFAULT_ROWS = 6
    _DEFAULT_COLS = 7
    _DEFAULT_MAX_ONNX_BATCH = 256


def export_onnx(
    model: Connect4Net,
    onnx_path: str,
    opset: int = _DEFAULT_OPSET,
) -> None:
    """Export `model` to ONNX with dynamic batch dim and named I/O."""
    model.eval()
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
        model,
        (dummy,),
        onnx_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes=dynamic_axes,
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
    p.add_argument("--opset", type=int, default=_DEFAULT_OPSET, help="ONNX opset version")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files")
    p.add_argument("--channels", type=int, default=_DEFAULT_CHANNELS)
    p.add_argument("--num-blocks", type=int, default=_DEFAULT_NUM_BLOCKS)
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="RNG seed for model initialization")
    args = p.parse_args()

    import random
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
