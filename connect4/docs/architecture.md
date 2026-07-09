# Architecture

This document describes the moving parts of the Connect 4 self-play pipeline:
the bitboard encoding, the **network-guided** MCTS, the self-play data
generator, the binary contract between Rust and Python, the network
architecture, and the loss.

If you only want to *use* the pipeline, the [README](../README.md) is enough.
This file is for understanding — and for changing the system in non-obvious
ways.

---

## 1. Bitboard encoding

The board is encoded in **column-major** order, with **7 bits per column** (6
data bits + 1 guard bit at the top of each column). Each player's pieces are
stored in a single `u64`. The two bitboards together fully describe the
position:

```
col 0 : bits  0..=6   (data 0..=5, guard 6)
col 1 : bits  7..=13
col 2 : bits 14..=20
col 3 : bits 21..=27
col 4 : bits 28..=34
col 5 : bits 35..=41
col 6 : bits 42..=48
```

So cell `(r, c)` (row `r` from 0 at the bottom, column `c` from 0 at the
left) maps to bit index `c * 7 + r`. The guard bit at `c * 7 + 6` is a
sentinel that's never set in a valid state — it exists to break the 4-shift
win check across column boundaries (see §1.2).

`BOARD_MASK` (in `bitboard.rs:56`) is the bitmask of all guard bits:

```text
BOARD_MASK = 0x0001_0204_0810_2040
            ─┬─ ─┬─ ─┬─ ─┬─ ─┬─ ─┬─ ─┬─
             │   │   │   │   │   │   └─ bit 6   (col 0 guard)
             │   │   │   │   │   └─── bit 13  (col 1 guard)
             │   │   │   │   └───── bit 20  (col 2 guard)
             │   │   │   └─────── bit 27  (col 3 guard)
             │   │   └────────── bit 34  (col 4 guard)
             │   └───────────── bit 41  (col 5 guard)
             └──────────────── bit 48  (col 6 guard)
```

Source: [`bitboard.rs:36-64`](../src_rust/src/bitboard.rs).

### 1.1 The "carry" trick for `next_empty_bit`

To find the lowest empty row in a column, the standard "add the bottom and
carry past occupied cells" trick is used. With the guard bit pre-set,
adding the bottom bit of the column rolls a carry up through the occupied
cells and lands on the first zero bit:

```text
occupied = (own | opp) & col_mask(c)
next     = ((occupied | guard_bit) + bottom_bit) & col_mask(c)
```

Source: [`bitboard.rs:73-85`](../src_rust/src/bitboard.rs).

### 1.2 The 4-shift win check

`check_win` looks for any 4 consecutive set bits along any of the four
alignment directions. The shifts are:

| Direction                | Shift amount | Why                                              |
|--------------------------|--------------|--------------------------------------------------|
| vertical (within column) | `1`          | one row up in the column-major layout            |
| horizontal (across cols) | `7`          | one column right = +7 bit-positions              |
| diagonal `\`             | `6`          | one col right (+7) and one row down (−1)         |
| diagonal `/`             | `8`          | one col right (+7) and one row up   (+1)         |

A single bitboard alignment of 4 consecutive bits in any direction is
detected by

$$
\text{win}(b) = (b \wedge (b \gg s)) \wedge (b \gg 2s) \wedge (b \gg 3s) \neq 0
$$

for each of the four shifts `s ∈ {1, 6, 7, 8}`. The guard bits at the top
of each column are stripped via `b & !BOARD_MASK` before the check
(defensive — valid play never sets them).

Source: [`bitboard.rs:88-117`](../src_rust/src/bitboard.rs).

### 1.3 Canonical perspective

Every board is stored with `own` = the **player to move**, and `opp` =
the opponent. After each `make_move`, the perspective is swapped:

```rust
self.own |= piece;
if check_win(self.own) { return Win; }
if self.ply() == 42 { return Draw; }
std::mem::swap(&mut self.own, &mut self.opp);
```

This means the network always sees the board from the current player's
point of view, with no extra turn plane needed for the policy to be
symmetric. Source: [`bitboard.rs:148-167`](../src_rust/src/bitboard.rs).

---

## 2. Network-guided Monte Carlo Tree Search

This is the AlphaZero MCTS: the network provides both the prior over
actions (used in PUCT) and the leaf value (no random rollouts).

### 2.1 PUCT selection

For a state `s` with children indexed by action `a`, the PUCT score is

$$
\text{PUCT}(s, a) \;=\; Q(s, a) \;+\; c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}
$$

where

- $Q(s, a) = W(s, a) / N(s, a)$ is the empirical mean of the leaf values
  seen from the perspective of the player to move at `s` after playing
  `a` (W = sum, N = count). Unvisited children use $Q = 0$.
- $P(s, a)$ is the prior probability of action `a` from the policy
  head of the network. The very first time we visit `s`, we call
  `network.evaluate(s)` to get $(P, V)$ and store $P$ on the children.
- $c_{\text{puct}} = 1.5$ is the exploration constant (matches AlphaZero).

Source: [`mcts.rs:343-372`](../src_rust/src/mcts.rs).

### 2.2 Leaf evaluation — no rollouts

At a never-before-seen state, we call `network.evaluate(s)` to get the
value $V(s)$ — the network's estimate of the position's outcome from the
current player's perspective. This is the **only** leaf evaluation: there
is no random rollout, no domain-specific playout, no Monte Carlo. The
value head is the entire evaluator.

$$
V(s) = f_{\text{value}}(s; \theta)
$$

where $f_{\text{value}}(\cdot; \theta)$ is the value head of the network
with parameters $\theta$. The training target for this head is the actual
game outcome $z \in \{-1, 0, +1\}$ (see §5), so the head learns to
predict the true win/loss/draw probability from any state.

This is the central AlphaZero insight: the network is both the prior and
the evaluator. The MCTS becomes a deterministic, value-aware, prior-aware
search over the policy space.

Source: [`mcts.rs:374-449`](../src_rust/src/mcts.rs).

### 2.3 Value convention and backup

The `search(node_idx)` function returns the value from the perspective of
the **player to move at `node_idx`**. So `+1` means "the player to move
here wins" and `−1` means "the player to move here loses".

When backing up, the value is flipped because the parent and the child
have different `own` players:

```text
parent.own  = player X
parent plays action a
  child.own = player Y  (opponent of X, after the swap)
  search(child) = v  (from Y's perspective)
back up at parent:  w[a] += -v
return to parent's parent:  -v
```

Two levels of flip → the value at the root is the correct ±1/0 outcome
for the player to move at the root. The "negation on backup" is the same
trick AlphaZero uses.

Source: [`mcts.rs:269-338`](../src_rust/src/mcts.rs).

### 2.4 Dirichlet noise at the root

To encourage exploration across independent games, the prior at the root
is mixed with Dirichlet noise:

$$
P'(s, a) = (1 - \varepsilon) \cdot P_{\text{network}}(a) + \varepsilon \cdot \eta_a,
\qquad \eta \sim \text{Dir}(\alpha)
$$

with $\alpha = 0.3$ and $\varepsilon = 0.25$ (defaults). The noise is
injected at the **start** of each MCTS run, so the search itself uses
the noised priors. Disable with `--no-noise` on the Rust CLI.

Source: [`mcts.rs:153-211`](../src_rust/src/mcts.rs).

### 2.5 Move policy

After `simulations` simulations, the visit counts are converted to a
move-sampling distribution:

$$
\pi(a \mid s) \;=\; \frac{N(s, a)^{1/\tau}}{\sum_{b} N(s, b)^{1/\tau}}
$$

where $\tau$ is the **temperature**. With $\tau = 1$ (default for
self-play) the policy is proportional to visit counts. With $\tau \to 0$
it collapses to $\arg\max_a N(s, a)$ — that's the mode used by the GUI.

Source: [`mcts.rs:213-269`](../src_rust/src/mcts.rs).

---

## 3. C4D1 binary format

The contract between the Rust data generator and the Python learner.
Documented here and enforced at both ends.

### 3.1 Layout

```
┌────────────────── HEADER (16 bytes) ──────────────────┐
│ 0x43 0x34 0x44 0x31  ← magic "C4D1"                   │
│ u32 LE  N            ← sample count                    │
│ 8 bytes  reserved    ← must be 0                       │
├────────────────── SAMPLE (56 bytes) × N ───────────────┤
│ u64 LE  own          ← current-player bitboard         │
│ u64 LE  opp          ← opponent bitboard               │
│ u64 LE  turn_mask    ← all 1s (constant bias plane)    │
│ 7 × f32 LE  π        ← MCTS policy                     │
│ f32 LE  z            ← game outcome ∈ {-1, 0, +1}      │
└────────────────────────────────────────────────────────┘
```

Total: `16 + 56 · N` bytes. The Python `dataset.py` reads this directly
via `np.fromfile` + `np.frombuffer`.

### 3.2 Why these fields

| Field      | Why                                                                                       |
|------------|-------------------------------------------------------------------------------------------|
| `own`      | Plane 0 of the network input (current player's pieces). 1 bit per cell, column-major.    |
| `opp`      | Plane 1 of the network input (opponent's pieces). Same layout.                            |
| `turn_mask`| Plane 2 of the network input — written as all 1s. Acts as a constant bias feature.       |
| `π`        | MCTS-improved target policy. 7 floats, sum to 1 over legal moves, 0 over illegal ones.   |
| `z`        | Target value, from THIS sample's perspective. `+1` if the player to move here wins, etc. |

The bit positions within `own` / `opp` are exactly the column-major layout
described in §1. The Rust side reconstructs the 3-plane tensor in
`board_to_input_array` (in `network.rs`) and the Python side does the
same in `decode_bitboard` (in `dataset.py`).

### 3.3 Value labeling

After a game of $M$ moves, we have $M$ recorded samples (one per move,
just before the move was made). The final outcome is:

- **Win**: the move that created the 4-in-a-row wins. The player to
  move at sample $M-1$ made the winning move → their value is $+1$.
  Every other sample's player to move lost → their value is $-1$.
- **Draw**: 42 moves, no winner. All values are $0$.

Source: [`main.rs:236-280`](../src_rust/src/main.rs).

---

## 4. Network architecture

A small ResNet, ~227K parameters. Designed to fit comfortably in the L2
cache of a 1650 and train in seconds.

```
input  (B, 3, 6, 7)
   │
   ├── input_conv: Conv2d(3 → 64, 3×3, padding=1) + BatchNorm + ReLU
   │
   ├── block 1: ResidualBlock(64)        ┐
   ├── block 2: ResidualBlock(64)        │  Trunk
   ├── block 3: ResidualBlock(64)        ┘
   │
   ├── policy_conv: Conv2d(64 → 2, 1×1) + BatchNorm + ReLU
   │      reshape to (B, 84)
   │      policy_fc: Linear(84 → 7)
   │      log_softmax
   │
   └── value_conv: Conv2d(64 → 1, 1×1) + BatchNorm + ReLU
          reshape to (B, 42)
          value_fc1: Linear(42 → 64) + ReLU
          value_fc2: Linear(64, 1) + tanh
```

`ResidualBlock` is the standard two-3×3-conv + skip + BatchNorm pattern
(He et al., 2015). All convs use padding=1 to preserve the 6×7
spatial size.

The model returns two outputs:

* `log_p`: (B, 7) log-probabilities over the 7 columns. The training
  loss is cross-entropy with the MCTS-improved target π.
* `v`: (B,) value in [-1, +1] (after `tanh`). The training loss is
  MSE against the game outcome z.

Source: [`model.py`](../src_python/model.py).

### 4.1 ONNX export

After training, `train.py` exports the model to ONNX with:

```python
torch.onnx.export(
    model, dummy, "connect4_model.onnx",
    input_names=["input"],
    output_names=["policy", "value"],
    dynamic_axes={"input": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
    opset_version=17,
)
```

The exported graph has:

| I/O      | Name    | Shape                  | Dtype |
|----------|---------|------------------------|-------|
| input    | input   | `(batch, 3, 6, 7)`     | f32   |
| output 0 | policy  | `(batch, 7)`           | f32   |
| output 1 | value   | `(batch,)`             | f32   |

The Rust side reads "policy" and "value" by name and softmaxes the
policy (since the model head outputs log-softmax, but the MCTS needs
probabilities for the PUCT priors). Source:
[`network.rs`](../src_rust/src/network.rs).

---

## 5. Loss

The total loss is a sum of two terms, both reduced as means over the
batch:

$$
L_{\text{total}} = L_{\text{value}} + L_{\text{policy}}
$$

### 5.1 Value loss

Standard MSE between the value head's prediction and the target:

$$
L_{\text{value}} = \frac{1}{B} \sum_{i=1}^{B} \left( v_\theta(s_i) - z_i \right)^2
$$

where $v_\theta(s_i) \in [-1, +1]$ is the network's prediction and
$z_i \in \{-1, 0, +1\}$ is the game outcome from sample $i$'s
perspective.

This is the only place the value head is trained. The MCTS uses the
value head as a leaf evaluator, but those values are *intermediate* —
they guide the search, they don't directly shape the loss. The
target for the value head is always the actual game outcome, not the
MCTS-improved Q.

### 5.2 Policy loss

Cross-entropy between the MCTS-improved policy and the network's
policy. Equivalently, the negative log-likelihood of the MCTS target
under the network's distribution:

$$
L_{\text{policy}} = - \frac{1}{B} \sum_{i=1}^{B} \sum_{a=1}^{7} \pi_i(a) \log p_\theta(a \mid s_i)
$$

where $\pi_i \in \Delta^6$ is the MCTS target and $p_\theta \in \Delta^6$
is the network's policy (after softmax). The constant $H(\pi)$ is dropped
from the KL formulation:

$$
L_{\text{policy}} = \text{KL}(\pi \parallel p) - H(\pi) = \sum_a \pi(a) \log \frac{\pi(a)}{p(a)} - \sum_a \pi(a) \log \pi(a)
$$

Minimising $L_{\text{policy}}$ ≡ minimising $\text{KL}(\pi \parallel p)$.

### 5.3 Total loss

$$
\boxed{L = \frac{1}{B} \sum_{i=1}^{B} \left[ \left( v_\theta(s_i) - z_i \right)^2 - \sum_{a=1}^{7} \pi_i(a) \log p_\theta(a \mid s_i) \right]}
$$

No L2 regularisation is added explicitly — `AdamW` with
`weight_decay=1e-4` handles it. Source: [`train.py:55-78`](../src_python/train.py).

### 5.4 Why this is the right loss

The pair $(v_\theta, p_\theta)$ is a learned model of the game from the
current player's perspective. Training them against the MCTS-improved
target means the network is trying to predict:

* $p(a | s)$: what move distribution would an *informed* search of this
  state produce?
* $v(s)$: what's the eventual win/loss/draw outcome from this state?

After one cycle of self-play, the network is "roughly calibrated" — its
priors are weak, its value predictions are noisy, but they correlate
with the truth. After many cycles, both improve and the MCTS becomes
sharper, generating better data, which trains a better network, etc.
This is the **bootstrapping** that makes AlphaZero work and that the
random-rollout variant in earlier versions of this project was missing.

---

## 6. The ONNX bridge

The pipeline needs the Rust MCTS to call the trained PyTorch model. We
bridge this by exporting the model to ONNX at the end of every training
cycle (`train.py`) and loading it in Rust via
[`tract-onnx`](https://github.com/sonos/tract), a pure-Rust ONNX runtime.

### 6.1 Why tract (and not `ort`)?

* **Pure Rust** — no system dependency on `libonnxruntime`, no install
  step. The whole stack compiles from `cargo build` alone.
* **`RunnableModel<TypedModel, TypedFact>` is `Send + Sync`**, so we
  wrap the session in `Arc` and share it across all rayon worker
  threads. Each worker calls `evaluate` on the shared `Network` and
  tract serialises the calls internally with optional intra-op
  parallelism (BLAS-backed GEMM/conv).
* **Our model is small** (~50K params, 3 conv blocks, 2 linear heads).
  CPU inference dominates the MCTS cost, not the NN cost.

### 6.2 Trade-off vs `ort` with CUDA

`ort`'s CUDA support is more mature than tract's. For our model on a
single 1650, CPU inference is plenty: ~50 µs per call on a modern
desktop, ~200 µs on a low-end laptop. If GPU inference becomes a
bottleneck, swap the `Network` body for an `ort`-backed implementation
— the `Eval` struct and `evaluate(&self, board)` API are stable, so the
MCTS doesn't need to change.

### 6.3 The Network wrapper

`src_rust/src/network.rs` defines:

```rust
pub struct Network {
    model: Option<Arc<OnnxModel>>,  // None → null network
}

pub struct Eval {
    pub policy: [f32; 7],
    pub value: f32,
}

impl Network {
    pub fn load(path: &Path) -> Result<Self, ...>;
    pub fn null() -> Self;
    pub fn evaluate(&self, board: Board) -> Eval;
    pub fn evaluate_batch(&self, boards: &[Board]) -> Vec<Eval>;
}
```

The `Network` is shared across all rayon workers via `Arc<Network>`. Each
worker calls `evaluate` on the shared reference; tract serialises the
calls internally.

### 6.4 The "null" network

`Network::null()` returns a Network with no model. Its `evaluate`
returns a uniform policy over legal moves and `value = 0`. This is a
**safety net** — the orchestrator's `init.py` ensures a real model
exists before the first self-play, so the null path is only hit if
the `.onnx` file was deleted manually or the cycle-0 bootstrap failed.

Source: [`network.rs`](../src_rust/src/network.rs).

---

## 7. Threading model

The self-play data generator spawns `num_games` independent games in
parallel via rayon. Each game has:

* **One MCTS instance** (per-game tree) — `MCTS::new` creates a fresh
  tree arena per game.
* **One RNG** — seeded deterministically from the master seed via
  SplitMix-style hashing (so the same master seed always produces the
  same games).
* **One `Arc<Network>` reference** — shared with all other games.

The `Network` is the only synchronisation point across threads.
tract's `RunnableModel::run` is internally serialised, but tract also
parallelises individual ops (e.g. the convs use BLAS-backed GEMM with
intra-op threads). On a 4-core / 8-thread CPU, the MCTS search and
the NN inference overlap nicely: while game $i$ is doing the PUCT
selection / back-up, game $i+1$ is in the middle of an NN forward
pass, and the BLAS threads are splitting the conv work for game $i$'s
forward. There is no per-thread lock on the Rust side.

This is the cleanest possible thread-safety story for a multi-game
MCTS:
* `Network::evaluate(&self, board)` is `&self` (immutable).
* `MCTS::run(&mut self, ...)` is per-MCTS-instance (no shared state).
* The MCTS tree is owned by the MCTS, not shared.
* The only shared state is the `Arc<Network>`, which is read-only.

Source: [`main.rs:140-180`](../src_rust/src/main.rs).

---

## 8. Cycle 0 bootstrap

The very first self-play has no trained model. We need *something* for
the MCTS to use. The orchestrator's `init.py` creates a random-init
`Connect4Net`, saves it as `connect4_model.pt`, and exports it to
`connect4_model.onnx`. The MCTS then uses this random model for
cycle 0.

Why a random model and not a null network? A random model has:

* **Non-uniform priors** (random conv weights produce some structure in
  the policy head output).
* **Non-zero values** (the value head's tanh output is some random
  signal in [-1, 1]).

These give the PUCT search a non-flat starting point. After one
training cycle, the model is no longer random — it's the first real
trained network. From cycle 1 onwards the network is the AlphaZero
loop.

The orchestrator's bootstrap step is idempotent: once
`connect4_model.onnx` exists, `init.py` is a no-op. The orchestrator
calls it on every cycle; only the first one does work.

Source: [`init.py`](../src_python/init.py).

---

## 9. Data flow

```text
   ┌──────────────── RUST (CPU) ──────────────────┐
   │                                              │
   │   load Arc<Network> from .onnx (tract)       │
   │                                              │
   │   rayon::par_iter                            │
   │     for game in 0..N:                        │
   │       Board = new()                          │
   │       loop:                                  │
   │         (π, q) = MCTS.run(board)              │
   │           MCTS = PUCT selection              │
   │             over (Q, N, P_network)           │
   │         record (own, opp, turn, π)           │
   │         sample action ~ π                    │
   │         apply make_move                      │
   │       end                                    │
   │       label values from outcome              │
   │   collect all samples into Vec               │
   │                                              │
   │   write C4D1 binary                          │
   └────────────────┬─────────────────────────────┘
                    │
                    ▼
                 selfplay.bin
                    │
                    ▼
   ┌──────────────── PYTHON (GPU) ────────────────┐
   │                                              │
   │   C4Dataset:                                 │
   │     read C4D1 header + samples               │
   │     decode bitboards → (N, 3, 6, 7)          │
   │                                              │
   │   train.py:                                  │
   │     AdamW + CosineAnnealingLR                │
   │     autocast(FP16) + GradScaler              │
   │     for batch in DataLoader:                 │
   │       log_p, v = model(planes)               │
   │       loss = MSE(v, z) + CE(π, log_p)        │
   │       backward, step, update                 │
   │     save .pt + export .onnx                  │
   │                                              │
   │   init.py (cycle 0 only):                    │
   │     create random-init model                 │
   │     save .pt + export .onnx                  │
   │                                              │
   │   gui.py:                                    │
   │     load state_dict                          │
   │     click column → forward → argmax          │
   │     animate piece                            │
   └──────────────────────────────────────────────┘
```

---

## 10. Source cross-references

| Topic                              | File / line range                                  |
|------------------------------------|----------------------------------------------------|
| Bitboard encoding                  | `src_rust/src/bitboard.rs:36-117`                  |
| `check_win` 4-shift trick          | `src_rust/src/bitboard.rs:88-117`                  |
| `next_empty_bit` carry trick       | `src_rust/src/bitboard.rs:73-85`                   |
| `make_move` + perspective swap     | `src_rust/src/bitboard.rs:148-167`                 |
| `Network` ONNX wrapper             | `src_rust/src/network.rs`                          |
| `Network::evaluate` (single state) | `src_rust/src/network.rs:board_eval`               |
| `Network::evaluate_batch`          | `src_rust/src/network.rs:batch_eval`               |
| `Network::null` fallback           | `src_rust/src/network.rs:null_eval`                |
| MCTS PUCT selection                | `src_rust/src/mcts.rs:343-372`                     |
| MCTS search + backup               | `src_rust/src/mcts.rs:269-338`                     |
| MCTS expand (network eval, no rollout) | `src_rust/src/mcts.rs:374-449`                  |
| Dirichlet noise at root            | `src_rust/src/mcts.rs:153-211`                     |
| Visit-count policy + temperature   | `src_rust/src/mcts.rs:213-269`                     |
| C4D1 binary writer                 | `src_rust/src/main.rs:295-323`                     |
| Self-play value labeling           | `src_rust/src/main.rs:236-280`                     |
| Network loading + Arc sharing      | `src_rust/src/main.rs:130-185`                     |
| C4D1 binary reader                 | `src_python/dataset.py:C4Dataset`                  |
| Bitboard → (3, 6, 7) decoder       | `src_python/dataset.py:decode_bitboard`            |
| `Connect4Net` architecture         | `src_python/model.py:Connect4Net`                  |
| Loss function                      | `src_python/train.py:compute_loss`                 |
| AMP + GradScaler training          | `src_python/train.py:main`                         |
| ONNX export (cycle N)              | `src_python/train.py:export_onnx`                  |
| ONNX bootstrap (cycle 0)           | `src_python/init.py`                               |
| GUI: AI move selection             | `src_python/gui.py:ai_select`                      |
| GUI: 4-shift win check (Python)    | `src_python/gui.py:has_win`                        |
| Pipeline orchestrator              | `run_pipeline.ps1`                                 |

---

## 11. References

- Silver, D. et al. (2017). *Mastering the Game of Go without Human
  Knowledge*. Nature 550, 354–359. The PUCT formula, value convention,
  loss equations, and the network-guided MCTS in this project are
  direct adaptations.
- He, K. et al. (2015). *Deep Residual Learning for Image Recognition*.
  The residual block in `model.py` follows this paper's bottleneck-free
  variant.
- ONNX: Open Neural Network Exchange format, [onnx.ai](https://onnx.ai).
- tract: pure-Rust ONNX runtime, [github.com/sonos/tract](https://github.com/sonos/tract).
