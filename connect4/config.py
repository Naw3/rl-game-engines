"""
config.py — Central configuration for the Connect4 AlphaZero pipeline.

Single source of truth for:
- Model Architecture
- MCTS Search & Parameters
- PyTorch Training Hyperparameters
- Binary Dataset Format (C4D1)
- GUI & Visual Options
- File Paths & Execution Devices
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class NetworkConfig:
    """Neural Network Architecture (Connect4Net)"""
    channels: int = 64
    num_blocks: int = 3
    input_planes: int = 3
    board_rows: int = 6
    board_cols: int = 7
    num_actions: int = 7


@dataclass
class MCTSConfig:
    """Rust MCTS Self-Play Search Parameters"""
    games: int = 32
    sims: int = 800
    cpu_batch_size: int = os.cpu_count() or 8
    gpu_batch_size: int = 32
    max_dispatcher_batch: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    seed: int = 42
    warmup: int = 20
    bench_iterations: int = 1000


@dataclass
class TrainConfig:
    """PyTorch Training Hyperparameters"""
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    replay_keep: int = 10
    num_workers: int = 2
    log_every: int = 20
    symmetry: bool = True
    use_amp: bool = True
    use_compile: bool = True


@dataclass
class DatasetConfig:
    """Binary C4D1 File Format Contract"""
    magic: str = "C4D1"
    header_size: int = 16
    sample_size: int = 56
    policy_size: int = 7
    onnx_opset: int = 18
    max_onnx_batch: int = 256


@dataclass
class GUIConfig:
    """Pygame & Console Interface Configuration"""
    window_w: int = 700
    window_h: int = 720
    board_top: int = 60
    fps: int = 60
    anim_frames: int = 12
    progress_bar_width: int = 20
    colors: dict = field(default_factory=lambda: {
        "bg": (24, 24, 36),
        "board": (40, 80, 200),
        "hole": (24, 24, 36),
        "red": (220, 50, 50),
        "yellow": (240, 210, 50),
        "grid": (200, 200, 240),
        "text": (240, 240, 240),
        "value_pos": (80, 200, 80),
        "value_neg": (200, 80, 80),
        "value_zero": (150, 150, 150),
    })


@dataclass
class PathConfig:
    """Project File Paths"""
    root: Path = PROJECT_ROOT
    model_pt: Path = PROJECT_ROOT / "connect4_model.pt"
    model_onnx: Path = PROJECT_ROOT / "connect4_model.onnx"
    selfplay_bin: Path = PROJECT_ROOT / "selfplay.bin"
    replay_dir: Path = PROJECT_ROOT / "replay"
    bench_temp_dir: Path = PROJECT_ROOT / ".bench_temp"


@dataclass
class DeviceConfig:
    """Hardware Execution Devices"""
    rust_device: str = "gpu"      # "gpu", "cpu", "auto"
    python_device: str = "cuda"   # "cuda", "cpu"


@dataclass
class BenchConfig:
    """Benchmark Specific Parameters (Isolated from main pipeline)"""
    games: int = 32
    sims: int = 800
    epochs: int = 5
    cpu_batch_size: int = os.cpu_count() or 8
    gpu_batch_size: int = 32
    train_batch_size: int = 256
    seed: int = 42


@dataclass
class PipelineConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)


# Global Singleton Configuration Instance
CONFIG = PipelineConfig()


def export_json() -> str:
    """Returns JSON string of CONFIG for Rust and external consumers."""
    import json
    return json.dumps({
        "network": asdict(CONFIG.network),
        "mcts": asdict(CONFIG.mcts),
        "train": asdict(CONFIG.train),
        "dataset": asdict(CONFIG.dataset),
        "gui": asdict(CONFIG.gui),
        "paths": {k: str(v) for k, v in asdict(CONFIG.paths).items()},
        "device": asdict(CONFIG.device),
        "bench": asdict(CONFIG.bench),
    }, indent=2)


def export_powershell_env() -> str:
    """Generates PowerShell $env: variable assignments for PowerShell scripts."""
    lines = [
        # Pipeline configs
        f'$env:GAMES = "{CONFIG.mcts.games}"',
        f'$env:SIMS = "{CONFIG.mcts.sims}"',
        f'$env:CPU_BATCH_SIZE = "{CONFIG.mcts.cpu_batch_size}"',
        f'$env:GPU_BATCH_SIZE = "{CONFIG.mcts.gpu_batch_size}"',
        f'$env:MAX_DISPATCHER_BATCH = "{CONFIG.mcts.max_dispatcher_batch}"',
        f'$env:EPOCHS = "{CONFIG.train.epochs}"',
        f'$env:TRAIN_BATCH_SIZE = "{CONFIG.train.batch_size}"',
        f'$env:REPLAY_KEEP = "{CONFIG.train.replay_keep}"',
        f'$env:NUM_WORKERS = "{CONFIG.train.num_workers}"',
        f'$env:LOG_EVERY = "{CONFIG.train.log_every}"',
        f'$env:MAX_GRAD_NORM = "{CONFIG.train.max_grad_norm}"',
        f'$env:SYMMETRY = "{1 if CONFIG.train.symmetry else 0}"',
        f'$env:ONNX_OPSET = "{CONFIG.dataset.onnx_opset}"',
        f'$env:RUST_DEVICE = "{CONFIG.device.rust_device}"',
        f'$env:PYTHON_DEVICE = "{CONFIG.device.python_device}"',
        f'$env:MODEL = "{CONFIG.paths.model_pt.name}"',
        f'$env:MODEL_ONNX = "{CONFIG.paths.model_onnx.name}"',
        f'$env:DATA = "{CONFIG.paths.selfplay_bin.name}"',
        
        # Bench configs (isolated)
        f'$env:BENCH_GAMES = "{CONFIG.bench.games}"',
        f'$env:BENCH_SIMS = "{CONFIG.bench.sims}"',
        f'$env:BENCH_EPOCHS = "{CONFIG.bench.epochs}"',
        f'$env:BENCH_CPU_BATCH_SIZE = "{CONFIG.bench.cpu_batch_size}"',
        f'$env:BENCH_GPU_BATCH_SIZE = "{CONFIG.bench.gpu_batch_size}"',
        f'$env:BENCH_TRAIN_BATCH_SIZE = "{CONFIG.bench.train_batch_size}"',
        f'$env:BENCH_SEED = "{CONFIG.bench.seed}"',
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if "--powershell" in sys.argv:
        print(export_powershell_env())
    else:
        print(export_json())
