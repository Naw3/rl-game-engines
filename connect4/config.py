import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class NetworkConfig:
    """Neural Network Architecture (Connect4Net)"""
    d_model: int = 64
    num_layers: int = 4
    nhead: int = 4
    input_planes: int = 3
    board_rows: int = 6
    board_cols: int = 7
    num_actions: int = 7


@dataclass
class MCTSConfig:
    """Rust MCTS Self-Play Search Parameters"""
    games: int = 256
    sims: int = 400
    # Playout Cap Randomization (PCR): mix a smaller and a full search
    # budget so the network learns from both cheap and deeply searched moves.
    pcr_full_probability: float = 0.25
    pcr_cheap_ratio: float = 0.1
    pcr_min_sims: int = 64
    cpu_batch_size: int = os.cpu_count() or 8
    gpu_batch_size: int = 32
    max_dispatcher_batch: int = 128
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
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    learning_rate_min: float = 1e-5
    lr_warmup_epochs: int = 5
    lr_schedule_epochs: int = 400
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    replay_keep: int = 10
    max_buffer_epochs: int = 0  # 0 = no pre-generation buffer (on-demand per epoch)
    num_workers: int = 0
    log_every: int = 0  # 0 = print only the epoch summary
    onnx_every: int = 1
    use_ema: bool = True
    ema_decay: float = 0.995
    symmetry: bool = True
    train_precision: str = "fp32"  # "fp32", "fp16", "bf16"
    infer_precision: str = "fp32"  # "fp32", "fp16", "int8"
    compile_mode: str = "reduce-overhead"  # "none", "default", "reduce-overhead", "max-autotune"
    channels_last: bool = False
    fused_adamw: bool = False
    prefetch_queue: int = 2
    confidence_loss_weight: float = 0.1

@dataclass
class DatasetConfig:
    """Binary C4D1 File Format Contract"""
    magic: str = "C4D1"
    header_size: int = 16
    sample_size: int = 60
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
    model_dir: Path = PROJECT_ROOT / "models"
    model_pt: Path = PROJECT_ROOT / "models" / "connect4_model.pt"
    model_ema_pt: Path = PROJECT_ROOT / "models" / "connect4_model_ema.pt"
    model_onnx: Path = PROJECT_ROOT / "models" / "connect4_model.onnx"
    model_ema_onnx: Path = PROJECT_ROOT / "models" / "connect4_model_ema.onnx"
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
    duration: int = 60
    sims: int = 800
    cpu_batch_size: int = os.cpu_count() or 8
    gpu_batch_size: int = 32
    train_batch_size: int = 2048
    seed: int = 42


@dataclass
class InferConfig:
    """Standalone AI Agent & GUI Inference Configuration"""
    infer_backend: str = "auto"  # "pytorch-cuda", "pytorch-cpu", "onnx-cuda", "tensorrt", "auto"
    sims: int = 0                 # 0 = confidence stop + max_think_time fallback
    max_think_time: float = 1.0   # Max search budget in seconds per move
    c_puct: float = 1.5           # PUCT exploration constant
    temperature: float = 0.0      # Action selection temperature
    batch_size: int = 32          # Fixed GPU leaf evaluation batch size
    device: str = "auto"          # Execution device ("cuda", "cpu", "auto")
    confidence_stop_enabled: bool = False
    confidence_threshold: float = 0.99
    confidence_min_sims: int = 0

    def __post_init__(self) -> None:
        self.infer_backend = self.infer_backend.strip().lower().replace("_", "-").replace(" ", "-")
        allowed = {"auto", "pytorch-cuda", "pytorch-cpu", "onnx-cuda", "tensorrt"}
        if self.infer_backend not in allowed:
            raise ValueError(
                f"Unsupported infer_backend={self.infer_backend!r}; "
                f"choose one of {', '.join(sorted(allowed))}."
            )


@dataclass
class PipelineConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)


# Global Singleton Configuration Instance
CONFIG = PipelineConfig()


def export_json() -> str:
    """Returns JSON string of CONFIG for Rust and external consumers."""
    return json.dumps({
        "network": asdict(CONFIG.network),
        "mcts": asdict(CONFIG.mcts),
        "train": asdict(CONFIG.train),
        "dataset": asdict(CONFIG.dataset),
        "gui": asdict(CONFIG.gui),
        "infer": asdict(CONFIG.infer),
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
        f'$env:PCR_FULL_PROBABILITY = "{CONFIG.mcts.pcr_full_probability}"',
        f'$env:PCR_CHEAP_RATIO = "{CONFIG.mcts.pcr_cheap_ratio}"',
        f'$env:PCR_MIN_SIMS = "{CONFIG.mcts.pcr_min_sims}"',
        f'$env:CPU_BATCH_SIZE = "{CONFIG.mcts.cpu_batch_size}"',
        f'$env:GPU_BATCH_SIZE = "{CONFIG.mcts.gpu_batch_size}"',
        f'$env:MAX_DISPATCHER_BATCH = "{CONFIG.mcts.max_dispatcher_batch}"',
        f'$env:C_PUCT = "{CONFIG.mcts.c_puct}"',
        f'$env:DIRICHLET_ALPHA = "{CONFIG.mcts.dirichlet_alpha}"',
        f'$env:DIRICHLET_EPSILON = "{CONFIG.mcts.dirichlet_epsilon}"',
        f'$env:TEMPERATURE = "{CONFIG.mcts.temperature}"',
        f'$env:SEED = "{CONFIG.mcts.seed}"',
        f'$env:EPOCHS = "{CONFIG.train.epochs}"',
        f'$env:TRAIN_BATCH_SIZE = "{CONFIG.train.batch_size}"',
        f'$env:LEARNING_RATE = "{CONFIG.train.learning_rate}"',
        f'$env:LEARNING_RATE_MIN = "{CONFIG.train.learning_rate_min}"',
        f'$env:LR_WARMUP_EPOCHS = "{CONFIG.train.lr_warmup_epochs}"',
        f'$env:LR_SCHEDULE_EPOCHS = "{CONFIG.train.lr_schedule_epochs}"',
        f'$env:WEIGHT_DECAY = "{CONFIG.train.weight_decay}"',
        f'$env:REPLAY_KEEP = "{CONFIG.train.replay_keep}"',
        f'$env:NUM_WORKERS = "{CONFIG.train.num_workers}"',
        f'$env:LOG_EVERY = "{CONFIG.train.log_every}"',
        f'$env:ONNX_EVERY = "{CONFIG.train.onnx_every}"',
        f'$env:MAX_GRAD_NORM = "{CONFIG.train.max_grad_norm}"',
        f'$env:SYMMETRY = "{1 if CONFIG.train.symmetry else 0}"',
        f'$env:COMPILE_MODE = "{CONFIG.train.compile_mode}"',
        f'$env:INFER_PRECISION = "{CONFIG.train.infer_precision}"',
        f'$env:CHANNELS_LAST = "{1 if CONFIG.train.channels_last else 0}"',
        f'$env:FUSED_ADAMW = "{1 if CONFIG.train.fused_adamw else 0}"',
        f'$env:ONNX_OPSET = "{CONFIG.dataset.onnx_opset}"',
        f'$env:MAX_THINK_TIME = "{CONFIG.infer.max_think_time}"',
        f'$env:INFER_BACKEND = "{CONFIG.infer.infer_backend}"',
        f'$env:CONFIDENCE_THRESHOLD = "{CONFIG.infer.confidence_threshold}"',
        f'$env:CONFIDENCE_STOP_ENABLED = "{1 if CONFIG.infer.confidence_stop_enabled else 0}"',
        f'$env:RUST_DEVICE = "{CONFIG.device.rust_device}"',
        f'$env:PYTHON_DEVICE = "{CONFIG.device.python_device}"',
        f'$env:MODEL = "{CONFIG.paths.model_pt.relative_to(PROJECT_ROOT)}"',
        f'$env:MODEL_ONNX = "{CONFIG.paths.model_onnx.relative_to(PROJECT_ROOT)}"',
        f'$env:DATA = "{CONFIG.paths.selfplay_bin.name}"',
        
        # Bench configs (isolated)
        f'$env:BENCH_DURATION = "{CONFIG.bench.duration}"',
        f'$env:BENCH_SIMS = "{CONFIG.bench.sims}"',
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
