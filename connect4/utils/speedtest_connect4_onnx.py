"""Measure the real Connect4 ONNX model through Rust GPU self-play.

This is intentionally separate from speedtest_onnx.py: it measures the full
CNN, quantize/dequantize overhead, MCTS batching and Rust dispatcher rather
than only the idealised DP4A matrix multiplication kernel.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INIT = ROOT / "src_python" / "init.py"
CARGO_MANIFEST = ROOT / "src_rust" / "Cargo.toml"
DURATION_SECONDS = 30
SIMULATIONS = 800
BATCH_SIZE = 64
SEED = 42


def run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(output.strip())
    return output


def main() -> None:
    print("Benchmark Connect4 réel : Rust + ONNX Runtime CUDA")
    print(
        f"Durée : {DURATION_SECONDS}s | simulations : {SIMULATIONS} | "
        f"batch GPU : {BATCH_SIZE} | seed : {SEED}"
    )
    results: dict[str, tuple[int, float, float]] = {}
    with tempfile.TemporaryDirectory(prefix="speedtest_connect4_", dir=ROOT) as temp:
        temp_dir = Path(temp)
        for precision in ("fp32", "int8"):
            model = temp_dir / f"connect4_{precision}.onnx"
            checkpoint = temp_dir / f"connect4_{precision}.pt"
            data = temp_dir / f"selfplay_{precision}.bin"
            run([
                str(PYTHON), str(INIT), "--force", "--seed", str(SEED),
                "--out-pt", str(checkpoint), "--out-onnx", str(model),
                "--infer-precision", precision,
            ])
            run([
                "cargo", "run", "--release", "--features", "cuda",
                "--manifest-path", str(CARGO_MANIFEST), "--",
                "--duration", str(DURATION_SECONDS), "--sims", str(SIMULATIONS),
                "--batch-size", str(BATCH_SIZE), "--output", str(data),
                "--model", str(model), "--device", "gpu", "--seed", str(SEED), "--verbose",
            ])
            games, games_per_sec, samples_per_sec = (data.with_suffix(data.suffix + ".stats")
                                                     .read_text(encoding="utf-8")
                                                     .strip().splitlines())
            results[precision] = (int(games), float(games_per_sec), float(samples_per_sec))
            print(
                f"{precision.upper()} : {games} parties | {float(games_per_sec):.1f} parties/s | "
                f"{float(samples_per_sec):.1f} samples/s"
            )

    fp32_games_per_sec = results["fp32"][1]
    int8_games_per_sec = results["int8"][1]
    print("\n--- Gain réel du pipeline Connect4 ---")
    print(f"Accélération INT8 : {int8_games_per_sec / fp32_games_per_sec:.2f}x")


if __name__ == "__main__":
    main()
