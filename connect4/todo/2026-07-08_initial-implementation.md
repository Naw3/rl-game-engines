---
status: in_progress
created: 2026-07-08
priority: high
tags: [mvp, bootstrap, full-stack]
---

# Initial full implementation of the Connect 4 AlphaZero-style pipeline

## Context

The user has laid out the full architecture for a self-play RL pipeline for Connect 4:

1. **Rust engine** (CPU-bound) — bitboard + MCTS + parallel self-play, writes binary training data
2. **Python learner** (GPU-bound) — CNN with policy/value heads, reads binary, trains, saves weights
3. **Orchestrator** — Rust → Python → cleanup → loop
4. **GUI** — Pygame showcase where the human plays against the latest model

The scaffold (`connect4/src_rust/`, `connect4/src_python/`, `Cargo.toml` with rayon/serde/bincode + release profile) exists but every source file is empty. This task fills them in end-to-end with a single coherent design so the pipeline can run from `cargo run --release` all the way through `python train.py` without any hand-holding.

## Design decisions (made without asking, will note in README)

- **Binary format "C4D1"** — version-tagged, 56 bytes/sample. Magic + count header, then (own, opp, turn-mask) u64s + 7×f32 policy + 1×f32 value. Documented in `docs/architecture.md` §3.
- **Bitboard layout** — column-major 7×7 with a guard bit at the top of each column. Standard 4-shift trick (`{1, 6, 7, 8}`) for win detection. Guard bits prevent cross-column false positives.
- **MCTS** — PUCT with `c_puct = 1.5`, 800 simulations/move during self-play, Dirichlet noise at root (α=0.3, ε=0.25). Final policy π = visit-count distribution.
- **CNN** — 3-plane input (own, opp, turn), 3× conv 3×3 (64 ch), policy head: 2-ch conv → FC(84, 7), value head: 1-ch conv → FC(42, 64) → FC(64, 1) + tanh.
- **Loss** — `(z − v)² + 0.5 · KL(π ∥ p) + L2`. KL reduced as `−π · log p` summed (since π is a target distribution, not a regularizer).
- **AMP** — FP16 autocast + GradScaler on the 1650 (sm_75, supports FP16 but not native BF16). This is the safe pick given the user's documented "FP16 numerical instability on FlowGen" experience — but for a tiny model with a stable target (z ∈ {−1, 0, 1}, π is a distribution), FP16 is fine.
- **GUI** — Pygame. Tkinter would be uglier for the falling-token animation.
- **Orchestrator** — `run_pipeline.sh` (the user's name) plus a `run_pipeline.ps1` companion for native Windows.

## Acceptance criteria

- [ ] `cargo check --release` in `src_rust/` passes with zero warnings.
- [ ] All four Python files `py_compile` cleanly.
- [ ] Rust can run a small self-play burst (e.g. 10 games) and write a valid `C4D1` file.
- [ ] Python can open that file, train for a few steps, and save a model.
- [ ] The GUI can load the model and play a move against a human.
- [ ] `run_pipeline.sh` and `run_pipeline.ps1` both run the cycle end-to-end (Rust → Python → cleanup → repeat).
- [ ] `README.md` documents the project at a high level.
- [ ] `docs/architecture.md` documents the math, the binary format, and the data flow with full equations (per user's doc-quality bar).
- [ ] `docs/architecture.md` cross-references the source files with `file:line` ranges.

## Plan

1. `bitboard.rs` — Board struct (own u64, opp u64), `make_move`, `check_win`, `get_legal_moves`, `is_terminal`, `key()` for transposition.
2. `mcts.rs` — Node, `search`, `expand`, PUCT selection, Dirichlet noise injection, visit-count policy extraction.
3. `main.rs` — Load (optional) Python-saved checkpoint, run N games in parallel via rayon, write C4D1 file.
4. `model.py` — `Connect4Net(nn.Module)` with shared trunk + policy/value heads, `save`/`load` matching Rust's expectations (we'll need to align state-dict keys, but Rust actually doesn't read the model — it just plays with MCTS from scratch; the only shared artifact is the binary data file).
5. `dataset.py` — `C4Dataset(Dataset)` that opens C4D1, mmap's, returns `(planes, policy, value)` tensors.
6. `train.py` — Model + dataset + Adam + GradScaler + torch.compile, save to `connect4_model.pt`.
7. `gui.py` — Pygame, click column → board.update → net forward → policy argmax → animate.
8. `run_pipeline.sh` and `run_pipeline.ps1` — infinite loop, catch failures, don't pile up temp files.
9. `README.md` + `docs/architecture.md`.
10. `cargo check` + `python -m py_compile` for verification.

## Notes

- The Rust side never reads the PyTorch model weights — it plays MCTS entirely from scratch using only Dirichlet + uniform priors during self-play. The network only sees the data Rust produces. This is a simplification of full AlphaZero (no network-guided MCTS during self-play data generation), but it's what the user's spec describes ("le MCTS joue des milliers de parties contre lui-même" + "Pour chaque coup joué, il fait tourner le MCTS"). If we later want guided MCTS during self-play, we can add a `tch` or `candle` inference path. Note this in the README.
- MCTS during self-play uses **uniform random priors** + Dirichlet noise at the root (standard approach when there's no network). After the first training cycle, we could plug the network in for priors — but the spec says MCTS plays against itself, so I'll keep it that way and document the upgrade path.
- The data file is written from many parallel rayon threads → contention. Use one writer thread (channel-based) so we serialize writes through a single mmap/buffer. This is the right call for both correctness and throughput.

## Re-open criteria

Pause this task if:
- `cargo check` fails in a way that requires redesigning the binary format
- The PUCT search produces obviously broken visit counts in a smoke test
- The user asks to change the data format or the loss function
