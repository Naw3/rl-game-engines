"""
utils/onnx_export.py — Shared ONNX export logic for init.py and train.py.

Both scripts need to export a Connect4Net to ONNX with identical settings.
This module centralises the export so neither script duplicates the logic.

Output contract (consumed by Rust network.rs via ORT):
    input  "input"  shape (batch, 3, 6, 7)  f32
    output "policy" shape (batch, 7)        f32  (log-probabilities)
    output "value"  shape (batch,)          f32  (in [-1, 1] via tanh)

The Rust side softmaxes the policy (since the model head outputs
log-softmax). The model is moved to CPU before export.
"""

from __future__ import annotations

import os
import sys
import logging
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
logging.getLogger("torch.onnx._internal").setLevel(logging.ERROR)
logging.getLogger("torch.export").setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from config import CONFIG
    _DEFAULT_PLANES = CONFIG.network.input_planes
    _DEFAULT_ROWS = CONFIG.network.board_rows
    _DEFAULT_COLS = CONFIG.network.board_cols
    _DEFAULT_OPSET = CONFIG.dataset.onnx_opset
    _DEFAULT_INFER_PRECISION = CONFIG.train.infer_precision
except Exception:
    _DEFAULT_PLANES = 3
    _DEFAULT_ROWS = 6
    _DEFAULT_COLS = 7
    _DEFAULT_OPSET = 18
    _DEFAULT_INFER_PRECISION = "fp32"


class _SuppressOutput:
    """Context manager that silences both Python-level and C-level stdout/stderr.

    torch.onnx.export prints verbose logs even when the Python logging level is
    set to ERROR. We redirect both the Python IO objects and the raw file
    descriptors 1 and 2 to /dev/null to catch everything.
    """

    def __enter__(self) -> "_SuppressOutput":
        self._py_stdout = sys.stdout
        self._py_stderr = sys.stderr
        self._devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = sys.stderr = self._devnull
        try:
            self._fd = os.open(os.devnull, os.O_WRONLY)
            self._saved_out = os.dup(1)
            self._saved_err = os.dup(2)
            os.dup2(self._fd, 1)
            os.dup2(self._fd, 2)
        except Exception:
            self._fd = None
        return self

    def __exit__(self, *_args: object) -> None:
        self._devnull.close()
        sys.stdout = self._py_stdout
        sys.stderr = self._py_stderr
        if self._fd is not None:
            try:
                os.dup2(self._saved_out, 1)
                os.dup2(self._saved_err, 2)
                os.close(self._fd)
                os.close(self._saved_out)
                os.close(self._saved_err)
            except Exception:
                pass


def export_onnx(
    model: "Connect4Net",  # noqa: F821 — avoid circular import at module level
    onnx_path: str | Path,
    opset: int = _DEFAULT_OPSET,
    infer_precision: str = _DEFAULT_INFER_PRECISION,
) -> None:
    """Export *model* to ONNX with dynamic batch dim and named I/O.

    Args:
        model:           A Connect4Net instance (trained or random-init).
        onnx_path:       Destination file path (will be created/overwritten).
        opset:           ONNX opset version (default from config, currently 18).
        infer_precision: ``"fp32"``, ``"fp16"``, or ``"int8"``.
                         ``"int8"`` applies dynamic quantisation via
                         onnxruntime after the initial export.

    The function always exports from CPU, so CUDA tensors are handled by
    copying weights to a temporary CPU model.
    """
    if infer_precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(
            f"Unsupported inference precision: {infer_precision!r}. "
            "Choose one of: fp32, fp16, int8."
        )

    # Always export on CPU — ORT reads the file, not CUDA tensors.
    if next(model.parameters()).device.type == "cuda":
        from model import Connect4Net  # local import to avoid circular dependency
        model_cpu = Connect4Net(channels=model.channels, num_blocks=model.num_blocks)
        model_cpu.load_state_dict(model.state_dict())
    else:
        model_cpu = model

    if infer_precision == "fp16":
        model_cpu = model_cpu.half()
        dummy = torch.randn(
            1, _DEFAULT_PLANES, _DEFAULT_ROWS, _DEFAULT_COLS, dtype=torch.float16
        )
    else:
        dummy = torch.randn(1, _DEFAULT_PLANES, _DEFAULT_ROWS, _DEFAULT_COLS)

    model_cpu.eval()

    dynamic_axes = {
        "input":  {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value":  {0: "batch_size"},
    }

    with _SuppressOutput():
        torch.onnx.export(
            model_cpu,
            (dummy,),
            str(onnx_path),
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    # Verify the exported graph is loadable.
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    if infer_precision == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import]

        # ORT's shape inference can conflict with certain model architectures;
        # clearing value_info before quantisation avoids those failures.
        quant_model = onnx.load(str(onnx_path))
        del quant_model.graph.value_info[:]
        onnx.save(quant_model, str(onnx_path))

        quantize_dynamic(
            str(onnx_path),
            str(onnx_path),
            weight_type=QuantType.QInt8,
            extra_options={"DisableShapeInference": True},
        )
        # Re-validate after quantisation.
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
