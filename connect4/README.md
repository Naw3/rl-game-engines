# Connect4 — Rust + PyTorch AlphaZero self-play pipeline

A from-scratch implementation of **real AlphaZero self-play for Connect 4**.
The Rust side runs network-guided MCTS in parallel on the CPU and writes
the (state, policy, value) training data to a compact binary file. The
Python side reads that file, trains a small CNN with PyTorch (policy + value
heads), exports the trained model to ONNX, and the next cycle's Rust MCTS
loads that ONNX to guide its search. A Pygame GUI lets you play against the
latest model.

The full training loop is:

```
┌────────────── RUST (CPU or GPU) ───────────────┐
│  load connect4_model.onnx                       │
│       │                                         │
│       ▼                                         │
│  MCTS × 800 sims per move,                      │
│   priors from policy head,                      │
│   leaf values from value head.                  │
│   backend: tract-onnx (CPU) or ort+CUDA (GPU)   │
│       │                                         │
│       ▼                                         │
│  write C4D1 binary (state, π, z)                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
                 selfplay.bin
                     │
                     ▼
┌──────────────── PYTHON (always GPU) ────────────┐
│  train.py:                                      │
│   read C4D1 → (B, 3, 6, 7) tensors              │
│   AMP FP16 + torch.compile                      │
│   loss = MSE(v, z) + CE(π, log p)               │
│   save .pt + export .onnx                       │
└────────────────────┬───────────────────────────┘
                     │
                     ▼
                connect4_model.pt
                connect4_model.onnx  ──► back to top
```

**Design contract:** the Python side is **always** trained on GPU
(CUDA). The Rust side is the variable — it can run on CPU (`tract`) or
GPU (`ort`+CUDA). This means the interesting benchmark is
`(py-gpu + rust-cpu)` vs `(py-gpu + rust-gpu)`, see the
[Benchmarking](#benchmarking--is-gpu-worth-it-for-rust) section.

The neural network is the *only* source of position evaluation — there is no
random rollout, no domain knowledge hard-coded into the search. The MCTS
select / expand / backup phases are all driven by the network's outputs.
That's the AlphaZero invariant this project preserves.

---

## Repository layout

```
connect4/
├── run_pipeline.ps1         # Endless self-play → train loop (PowerShell)
├── src_rust/                # The network-guided MCTS engine (CPU)
│   ├── Cargo.toml
│   └── src/
│       ├── bitboard.rs      # Board, move, win — pure bit math
│       ├── network.rs       # ONNX inference wrapper (tract)
│       ├── mcts.rs          # PUCT search, Dirichlet noise, network eval
│       └── main.rs          # Parallel self-play, C4D1 writer
├── src_python/              # The learner + GUI (GPU)
│   ├── requirements.txt
│   ├── model.py             # Connect4Net: small ResNet, policy + value heads
│   ├── dataset.py           # Reads the C4D1 binary
│   ├── train.py             # AMP FP16 + torch.compile, saves .pt + .onnx
│   ├── init.py              # Random-init bootstrap for cycle 0
│   └── gui.py               # Pygame interface: play vs the model
├── docs/
│   └── architecture.md      # Full math + binary format + data flow
├── todo/                    # Per-task planning files
│   └── 2026-07-08_initial-implementation.md
└── README.md                # You are here
```

---

## Quick start

### One-time setup

```bash
# Rust toolchain (stable).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Python deps (CUDA build of PyTorch; for CPU-only, change the index URL).
python -m pip install -r src_python/requirements.txt
```

### The one-command loop (Windows PowerShell)

```powershell
.\run_pipeline.ps1
```

The very first cycle does an extra bootstrap step (`init.py`) that creates a
random-init model and exports it to ONNX. After that the loop is:

1. Rust spawns `num_games` self-play games in parallel (rayon).
2. Each game plays out with 800 network-guided MCTS simulations per move.
   The network's policy head provides the PUCT priors; the value head
   provides the leaf value (no rollouts).
3. The (state, MCTS policy, outcome) triples are written to `selfplay.bin`
   in the [C4D1 format](docs/architecture.md#3-c4d1-binary-format).
4. Python reads `selfplay.bin`, trains `connect4_model.pt` for 5 epochs,
   re-exports `connect4_model.onnx`.
5. `selfplay.bin` is deleted and the cycle restarts.

Useful env-var overrides:

```powershell
$env:GAMES = 128; $env:SIMS = 1600; $env:EPOCHS = 10; $env:BATCH = 512; $env:RUST_DEVICE = "gpu"; .\run_pipeline.ps1
```

Available env vars:

| Var             | Default | Notes                                                              |
|-----------------|---------|--------------------------------------------------------------------|
| `GAMES`         | `64`    | self-play games per cycle                                          |
| `SIMS`          | `800`   | MCTS simulations per move                                          |
| `EPOCHS`        | `5`     | training epochs per cycle                                          |
| `BATCH`         | `256`   | training batch size                                                |
| `DATA`          | `selfplay.bin` | path to the C4D1 dataset file                              |
| `MODEL`         | `connect4_model.pt`    | output path for the trained `.pt`              |
| `MODEL_ONNX`    | `connect4_model.onnx`  | output path for the ONNX export                |
| `SLEEP`         | `2`     | seconds between cycles                                             |
| `CARGO`         | `cargo` | cargo binary path                                                  |
| `PYTHON`        | `python`| python binary path                                                 |
| `RUST_DEVICE`   | `auto`  | Rust inference backend: `cpu` \| `gpu` \| `auto`                   |
| `PYTHON_DEVICE` | `cuda`  | Python training device: `cuda` (always GPU) \| `cpu` (debug only)  |
| `DEVICE`        | —       | Legacy. If set, both `RUST_DEVICE` and `PYTHON_DEVICE` inherit it. |

The split exists because the Python side is **always GPU** by design —
varying it isn't useful and CUDA training is dramatically faster than CPU.
Only the Rust side varies between CPU (`tract`) and GPU (`ort`+CUDA).

### Play against the model

Once you have a trained `connect4_model.pt`:

```bash
python src_python/gui.py --model connect4_model.pt
```

The human plays Red, the AI plays Yellow. Left-click to drop a piece. Press
`R` to reset, `Q` to quit.

### Step-by-step (for debugging)

```powershell
# 0. Bootstrap a random-init model + ONNX export.
Push-Location src_python
python init.py
Pop-Location

# 1. Generate training data (network-guided MCTS).
Push-Location src_rust
cargo run --release -- -g 64 -s 800 -o ..\selfplay.bin -v
Pop-Location

# 2. Train + ONNX re-export.
Push-Location src_python
python train.py --data ..\selfplay.bin --out ..\connect4_model.pt
Pop-Location

# 3. Play.
python gui.py --model ..\connect4_model.pt
```

---

## How it works (in one paragraph)

For every state, the MCTS does 800 PUCT-guided simulations. On each
expansion, the Rust MCTS calls the ONNX-exported PyTorch model (in-process,
via `tract-onnx`) to get the prior over actions (used in the PUCT formula)
and the value of the new leaf. After all simulations, the visit counts are
converted to a policy distribution π. After each game, every recorded
state is labeled with the final outcome z ∈ {−1, 0, +1} from the
perspective of the player to move at that state. The Python side trains
a small CNN to predict (π, z) given the 3-plane board representation, using
cross-entropy on the policy and MSE on the value. The trained model is
re-exported to ONNX and the cycle restarts.

For the math, the binary format, and the data flow, see
[`docs/architecture.md`](docs/architecture.md).

---

## MCTS defaults (all tunable in `mcts.rs` / CLI)

| Parameter              | Default | Notes                                    |
|------------------------|---------|------------------------------------------|
| `simulations` / move   | 800     | `-s/--sims` on the Rust CLI              |
| `c_puct`               | 1.5     | Exploration constant                     |
| `dirichlet_alpha`      | 0.3     | α for root noise                         |
| `dirichlet_epsilon`    | 0.25    | ε for root noise (set 0 via `--no-noise`)|
| `temperature`          | 1.0     | Move sampling; 0 = greedy                |
| `num_games`            | 64      | `-g/--games`                             |

## Training defaults (all tunable via CLI in `train.py`)

| Parameter       | Default | Notes                                          |
|-----------------|---------|------------------------------------------------|
| `epochs`        | 5       | `--epochs`                                     |
| `batch`         | 256     | `--batch`                                      |
| `lr`            | 1e-3    | `--lr`                                         |
| `weight_decay`  | 1e-4    | `--weight-decay`                               |
| `amp`           | on      | `--no-amp` to disable (e.g. for FP16 debugging)|
| `torch.compile` | on      | `--no-compile` if your PT version is < 2.0     |
| `onnx export`   | on      | `--no-onnx` to skip                            |

---

## The ONNX bridge — dual backend

The pipeline needs the Rust MCTS to call the trained PyTorch model. We
bridge this by exporting the model to ONNX at the end of every training
cycle (`train.py`) and loading it in Rust. **Two inference backends are
supported** so the user can pick the right tool per scenario:

* **`tract-onnx`** (CPU): pure Rust, no FFI, no external install.
  ~50 µs/call on a modern desktop, ~200 µs on a low-end laptop.
  Fastest on small models like ours (227K params) because there's
  no FFI marshalling overhead. Always available.

* **`ort`** (GPU/CPU): FFI to Microsoft onnxruntime. Supports CUDA
  via the CUDA execution provider. Needs `--features cuda` at build
  time + `onnnxruntime-gpu` installed at runtime.

The Rust-side choice is made at runtime via the `--device` flag (driven
by the `RUST_DEVICE` env var):

| `RUST_DEVICE` | Rust backend                | Rust build flag                  |
|---------------|-----------------------------|----------------------------------|
| `cpu`         | `tract-onnx` (pure Rust)    | `cargo build --release`          |
| `gpu`         | `ort` + CUDA EP             | `cargo build --release --features cuda` |
| `auto`        | GPU if compiled-in & init OK, else CPU | `--features cuda` (graceful fallback) |

The Python side is **always** trained on CUDA (when CUDA is available;
falls back to CPU with a clear warning). This is hard-coded by design —
varying Python's device isn't useful and CUDA training is ~10× faster
than CPU on this model size. Use `train.py --cpu` directly only for
debugging.

The orchestrator's `RUST_DEVICE` and `PYTHON_DEVICE` env vars drive the
two sides independently — see the [env vars](#available-env-vars) table
above. The legacy single `DEVICE` env var still works (both sides
inherit it) for backwards compat.

### CUDA setup (only if you want `--device gpu`)

```bash
# 1. Install CUDA toolkit 12.x (~3 GB, 10 min)
# 2. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 3. Install onnxruntime-gpu
pip install onnxruntime-gpu
# 4. Rebuild Rust with the cuda feature
cd src_rust
cargo build --release --features cuda
```

If you skip this, `--device gpu` will fail at runtime with a clear
error. `--device cpu` and `--device auto` work out of the box.

### Benchmarking — is GPU worth it for Rust?

The question we actually want to answer is: **does moving Rust self-play
from CPU to GPU speed up a full training cycle** (self-play → train → ONNX
export), with Python training held constant on CUDA?

We compare two configurations:

| Config | Rust self-play | Python training |
|--------|----------------|-----------------|
| **A**  | CPU (`tract`)  | CUDA            |
| **B**  | GPU (`ort`+CUDA) | CUDA          |

Each "cycle" = one full run of the orchestrator (self-play + train +
ONNX export). Use `Measure-Command` to time a cycle in each config and
compare wall-clock.

```powershell
# Config A: Rust CPU + Python CUDA (no CUDA setup needed beyond PyTorch CUDA build)
$env:RUST_DEVICE = "cpu"
$env:GAMES = 64; $env:SIMS = 800; $env:EPOCHS = 5
$timeA = Measure-Command { .\run_pipeline.ps1 }
# Ctrl-C after the first cycle completes (or set SLEEP=0 and add a cycle limit)

# Config B: Rust GPU + Python CUDA (needs `cargo build --release --features cuda`)
$env:RUST_DEVICE = "gpu"
$timeB = Measure-Command { .\run_pipeline.ps1 }

# Compare
"Config A (rust-cpu):  $($timeA.TotalSeconds) s"
"Config B (rust-gpu):  $($timeB.TotalSeconds) s"
"Speedup:             $([math]::Round($timeA.TotalSeconds / $timeB.TotalSeconds, 2))x"
```

The script reports `self-play done in X.X s` and `train done in X.X s`
per cycle, so you can also see the breakdown within a single cycle.

For pure NN-inference micro-benchmarks (no MCTS, no training), the Rust
binary has a `benchmark` subcommand:

```powershell
# CPU inference baseline (no CUDA needed)
cargo run --release --manifest-path src_rust/Cargo.toml -- benchmark -n 5000 -d cpu

# GPU inference (after `cargo build --release --features cuda`)
cargo run --release --features cuda --manifest-path src_rust/Cargo.toml -- benchmark -n 5000 -d gpu
```

Output is mean µs/call and throughput (calls/sec). For the 227K-param
model, typical results on a 1650 are ~2-3× speedup on raw NN inference.
The end-to-end speedup is smaller because MCTS overhead (board copy,
PUCT bookkeeping, win checks) is CPU-bound and dominates ~20% of the
wall time.

The ONNX graph contract is fixed:

* input `"input"` shape `(batch, 3, 6, 7)` f32
* output `"policy"` shape `(batch, 7)` f32 — log-probabilities
* output `"value"` shape `(batch,)` f32 — in [-1, 1] via tanh

`init.py` and `train.py` both use `dynamic_axes={"input": {0: "batch"}}`
so the same model handles any batch size at inference.

## Cycle 0 bootstrap

The very first self-play run has no trained model. We need *something*
for the MCTS to use as a prior and value source. `init.py` creates a
random-init `Connect4Net`, saves it as `connect4_model.pt`, and exports
it as `connect4_model.onnx`. The MCTS uses this random model for cycle
0 (which still beats uniform priors + value=0 because the random network
produces some structure). Cycle 0's dataset is then used to train the
first real model. From cycle 1 onwards the network is the AlphaZero loop.

The orchestrator detects missing `connect4_model.onnx` automatically and
runs `init.py` once. After that the step is a no-op (idempotent).

---

## Caveats

* **No replay buffer across cycles.** Each cycle's training only sees that
  cycle's games. A real AlphaZero keeps a sliding window of the most recent
  ~1M positions. Easy to add: append the current `selfplay.bin` to a buffer
  file instead of deleting it.
* **No symmetry augmentation.** Connect 4 has a left-right symmetry that we
  don't exploit; doubling the effective dataset is a 5-line change in
  `dataset.py`.
* **No evaluation against a baseline.** Strength is judged by "did the loss
  go down" and "does the GUI play sensibly".
* **Python training is fixed on CUDA.** Use `train.py --cpu` to override
  the device (debugging only). This is a design choice: CUDA training is
  ~10× faster and varying Python's device isn't useful.

## Upgrade paths (in order of expected impact)

1. **Replay buffer** — keep the last N cycles of self-play data, weighted
   toward the most recent. Expected effect: more sample-efficient training.
2. **Batched NN evaluation** in MCTS — collect unique states to evaluate,
   flush as a single batch. tract's batched path is much faster than
   per-state calls.
3. **Symmetry augmentation** in `dataset.py` (horizontal flip).
4. **Periodic evaluation** against a fixed baseline (random player or the
   model from K cycles ago) to track real strength.
5. **Larger model** — current 64-channel ResNet is ~50K params. Scaling to
   128–256 channels is a 2-line change and should help.
6. **Batched NN inference on GPU** — already supported via `ort`+CUDA EP
   with `--features cuda` + `RUST_DEVICE=gpu`. Future win: call the
   session with `batch_size > 1` instead of per-state calls.

---

## License

This is a personal portfolio / learning project; use it however you want.
The mathematical formulations and loss equations follow Silver et al. 2017
(*Mastering the Game of Go without Human Knowledge*).
