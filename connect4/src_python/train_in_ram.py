import argparse
import subprocess
import sys
from pathlib import Path
from dataset import C4Dataset

def run_selfplay_in_ram(exe_path: str, duration: int, model_path: str, seed: int, batch_size: int, device: str) -> bytes:
    """Run the Rust selfplay executable and capture the dataset from stdout."""
    cmd = [
        exe_path,
        "--duration", str(duration),
        "--sims", "800",
        "--batch-size", str(batch_size),
        "--model", model_path,
        "--device", device,
        "--seed", str(seed),
        "-o", "-",  # Output to stdout!
        "--verbose"
    ]
    
    print(f"[python] Starting Rust MCTS in RAM (duration={duration}s)...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Read stdout and stderr
    stdout_data, stderr_data = process.communicate()
    
    # Print the stderr (which contains the progress bars and stats)
    sys.stderr.write(stderr_data.decode("utf-8", errors="replace"))
    
    if process.returncode != 0:
        raise RuntimeError(f"Rust MCTS failed with exit code {process.returncode}")
        
    return stdout_data

def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero model completely in RAM without disk I/O for selfplay.bin")
    parser.add_argument("--exe", type=str, default="../src_rust/target/release/connect4_mcts.exe", help="Path to Rust executable")
    parser.add_argument("--model", type=str, default="../models/test_fp32.onnx", help="Path to ONNX model")
    parser.add_argument("--duration", type=int, default=10, help="Duration of selfplay in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="gpu", help="Device (cpu/gpu/auto)")
    
    args = parser.parse_args()
    
    # 1. Generate data entirely in RAM
    raw_bytes = run_selfplay_in_ram(
        args.exe, args.duration, args.model, args.seed, args.batch_size, args.device
    )
    
    print(f"\n[python] Received {len(raw_bytes) / 1024 / 1024:.2f} MB of data in RAM.")
    
    # 2. Parse it directly into the PyTorch dataset
    dataset = C4Dataset(raw_bytes)
    print(f"[python] Loaded dataset with {len(dataset)} samples in RAM.")
    
    # Now you can pass `dataset` to torch.utils.data.DataLoader and train!
    print("[python] Ready for training step...")

if __name__ == "__main__":
    main()
