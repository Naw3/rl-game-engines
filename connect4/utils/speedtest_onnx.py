"""Reproduce speedtest_pytorch.py with ONNX Runtime CUDA through Rust.

It benchmarks 100 4096x4096 matrix multiplications. Inputs are bound once by
the Rust IO-binding benchmark and kept on CUDA; export/load/warmup are excluded.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
CARGO_MANIFEST = ROOT / "src_rust" / "Cargo.toml"
N = 4096
ITERATIONS = 100
WARMUP = 20


def make_model(path: Path, precision: str) -> None:
    if precision == "fp32":
        op_type, output_type = "MatMul", TensorProto.FLOAT
        nodes = [helper.make_node(op_type, ["a", "b"], ["y"])]
    elif precision == "fp16":
        op_type, output_type = "MatMul", TensorProto.FLOAT16
        nodes = [
            helper.make_node("Cast", ["a"], ["a16"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["b"], ["b16"], to=TensorProto.FLOAT16),
            helper.make_node(op_type, ["a16", "b16"], ["y"]),
        ]
    elif precision == "int8":
        op_type, output_type = "MatMulInteger", TensorProto.INT32
        nodes = [
            helper.make_node("Cast", ["a"], ["a8"], to=TensorProto.INT8),
            helper.make_node("Cast", ["b"], ["b8"], to=TensorProto.INT8),
            helper.make_node(op_type, ["a8", "b8"], ["y"]),
        ]
    else:
        raise ValueError(f"Precision inconnue : {precision}")

    graph = helper.make_graph(
        nodes,
        f"matmul_{precision}",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [N, N]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [N, N]),
        ],
        [helper.make_tensor_value_info("y", output_type, [N, N])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run_rust(model: Path) -> float:
    command = [
        "cargo", "run", "--release", "--features", "cuda",
        "--manifest-path", str(CARGO_MANIFEST), "--",
        "matmul-benchmark", "--model", str(model),
        "--iterations", str(ITERATIONS), "--warmup", str(WARMUP),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(output.strip())
    match = re.search(r"mean:\s+([0-9.]+)\s+(?:µs|Âµs)/call", output)
    if not match:
        raise RuntimeError(f"Impossible de lire le résultat Rust :\n{output}")
    return float(match.group(1))


def main() -> None:
    print("Benchmark ONNX Runtime via Rust ort (CUDAExecutionProvider)")
    print(f"Matrice : {N}x{N} | itérations : {ITERATIONS} | warmup : {WARMUP}")
    results: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="speedtest_onnx_", dir=ROOT) as temp:
        temp_dir = Path(temp)
        for precision in ("fp32", "fp16", "int8"):
            model = temp_dir / f"matmul_{precision}.onnx"
            make_model(model, precision)
            mean_us = run_rust(model)
            results[precision] = mean_us
            print(f"Temps {precision.upper()} : {mean_us * ITERATIONS / 1_000_000:.4f} s")

    print("\n--- Ratios de vitesse (par rapport au FP32) ---")
    for precision in ("fp16", "int8"):
        print(f"Accélération {precision.upper()} : {results['fp32'] / results[precision]:.2f}x")


if __name__ == "__main__":
    main()
