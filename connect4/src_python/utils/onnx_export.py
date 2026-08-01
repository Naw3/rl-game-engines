"""
utils/onnx_export.py — Shared ONNX export logic for init.py and train.py.

Both scripts need to export a Connect4Net to ONNX with identical settings.
This module centralises the export so neither script duplicates the logic.

Output contract (consumed by Rust network.rs via ORT):
    input  "input"      shape (batch, 3, 6, 7)  f32
    output "policy"     shape (batch, 7)        f32  (log-probabilities)
    output "value"      shape (batch,)          f32  (in [-1, 1] via tanh)
    output "moves_left" shape (batch,)          f32

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
    onnx_path: str | Path | Any,
    opset: int = _DEFAULT_OPSET,
    infer_precision: str = _DEFAULT_INFER_PRECISION,
    calibration_samples: "numpy.ndarray | None" = None,
    verify: bool = False,
) -> None:
    """Export *model* to ONNX with dynamic batch dim and named I/O.

    Args:
        model:           A Connect4Net instance (trained or random-init).
        onnx_path:       Destination file path (str/Path) or BytesIO buffer.
        opset:           ONNX opset version (default from config, currently 18).
        infer_precision: ``"fp32"``, ``"fp16"``, or ``"int8"``.
                         ``"int8"`` applies dynamic quantisation via
                         onnxruntime after the initial export.
        verify:          If True, run onnx.checker.check_model on the exported graph.

    The function always exports from CPU, so CUDA tensors are handled by
    copying weights to a temporary CPU model.
    """
    if infer_precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(
            f"Unsupported inference precision: {infer_precision!r}. "
            "Choose one of: fp32, fp16, int8."
        )

    # Fast path: if the ONNX file already exists, update weights directly in <10ms
    if infer_precision == "fp32" and isinstance(onnx_path, (str, Path)) and update_onnx_weights(model, onnx_path):
        return

    if isinstance(onnx_path, (str, Path)):
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        export_target: Any = str(onnx_path)
    else:
        export_target = onnx_path

    # Always export on CPU — ORT reads the file, not CUDA tensors.
    import copy
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_cpu = copy.deepcopy(inner).cpu()

    if infer_precision == "fp16":
        model_cpu = model_cpu.half()
        dummy = torch.randn(
            2, _DEFAULT_PLANES, _DEFAULT_ROWS, _DEFAULT_COLS, dtype=torch.float16
        )
    else:
        dummy = torch.randn(2, _DEFAULT_PLANES, _DEFAULT_ROWS, _DEFAULT_COLS)

    model_cpu.eval()

    dynamic_axes = {
        "input":  {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value":  {0: "batch_size"},
        "moves_left": {0: "batch_size"},
    }

    with _SuppressOutput():
        torch.onnx.export(
            model_cpu,
            (dummy,),
            export_target,
            input_names=["input"],
            output_names=["policy", "value", "moves_left"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=False,
            export_params=True,
            keep_initializers_as_inputs=True,
        )

    if verify and isinstance(onnx_path, (str, Path)):
        # Verify the exported graph is loadable when explicitly requested.
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)


def update_onnx_weights(model: torch.nn.Module, onnx_path: str | Path) -> bool:
    """Fast inline weight updater for ONNX models without running full PyTorch JIT tracing.

    Returns True if weights were updated in-memory (<10ms), False if fallback export is required.
    """
    import os, onnx
    from onnx import numpy_helper

    path_str = str(onnx_path)
    if not os.path.exists(path_str):
        return False

    try:
        onnx_model = onnx.load(path_str)
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = inner.state_dict()

        dict_by_name = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
        dict_by_shape = {}
        for k, v in state_dict.items():
            dict_by_shape.setdefault(tuple(v.shape), []).append(v.detach().cpu().numpy())
        
        shape_ptrs = {shape: 0 for shape in dict_by_shape}

        updated_count = 0
        for init in onnx_model.graph.initializer:
            key = init.name
            init_shape = tuple(init.dims)

            if key in dict_by_name:
                tensor_np = dict_by_name[key]
            elif init_shape in dict_by_shape and shape_ptrs[init_shape] < len(dict_by_shape[init_shape]):
                idx = shape_ptrs[init_shape]
                tensor_np = dict_by_shape[init_shape][idx]
                shape_ptrs[init_shape] += 1
            else:
                continue

            new_init = numpy_helper.from_array(tensor_np, name=key)
            init.CopyFrom(new_init)
            updated_count += 1

        if updated_count == 0:
            return False

        onnx.save_model(onnx_model, path_str, save_as_external_data=False)
        return True
    except Exception:
        return False

    if infer_precision == "int8":
        import logging
        from onnxruntime.quantization import QuantType, quantize_static, QuantFormat, CalibrationDataReader  # type: ignore[import]
        
        class Connect4CalibrationReader(CalibrationDataReader):
            def __init__(self, samples_np):
                self.enum_data = iter([{"input": samples_np}])
            def get_next(self):
                return next(self.enum_data, None)

        if calibration_samples is None:
            # Fallback random calibration for init.py
            import numpy as np
            calibration_samples = np.random.randn(32, _DEFAULT_PLANES, _DEFAULT_ROWS, _DEFAULT_COLS).astype(np.float32)

        reader = Connect4CalibrationReader(calibration_samples)
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        with _SuppressOutput():
            try:
                quantize_static(
                    onnx_path,
                    onnx_path,
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QOperator,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                    nodes_to_exclude=["node_linear_17", "node_masked_fill"],
                    extra_options={"DisableShapeInference": True},
                )
            except Exception:
                pass
        logger.setLevel(old_level)
        
        # Re-validate after quantisation.
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
