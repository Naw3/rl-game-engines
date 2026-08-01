"""
selfplay.py — In-process GPU self-play for Connect4 AlphaZero training.

Eliminates all disk I/O by generating training data directly in RAM.
A background thread continuously plays games using MCTS + the current model,
pushing (planes, policy, value) samples into a thread-safe ReplayBuffer.
The training loop pulls samples from this buffer each epoch.

Architecture:
    ┌─────────────────────┐      ┌──────────────────────┐
    │  Training Thread    │      │  Self-Play Thread     │
    │  (forward/backward) │      │  (MCTS + inference)   │
    │                     │      │                       │
    │  model.train()      │      │  sp_model.eval()      │
    │  loss.backward()    │      │  torch.no_grad()      │
    │  optimizer.step()   │      │  play_batch() loop    │
    └────────┬────────────┘      └──────────┬────────────┘
             │    ┌──────────────────┐       │
             └────► ReplayBuffer     ◄───────┘
                  │ (thread-safe     │
                  │  circular deque) │
                  └──────────────────┘
"""

from __future__ import annotations

import copy
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# ─── Constants ──────────────────────────────────────────────────────────

ROWS, COLS = 6, 7

# Precomputed bit masks: column c occupies bits [c*7 .. c*7+5]. 6 bits
# per column (rows 0-5). Bit c*7+6 is a gap/sentinel never set.
_COL_FULL = tuple(0x3F << (c * 7) for c in range(COLS))

# (6, 7) table: _BIT_TABLE[r, c] = 1 << (c*7 + r), for vectorized
# bitboard → plane conversion.
_BIT_TABLE = np.array(
    [[1 << (c * 7 + r) for c in range(COLS)] for r in range(ROWS)],
    dtype=np.uint64,
)


# ─── Board helpers (pure functions on bitboard ints) ────────────────────

def _check_win(bb: int) -> bool:
    """Bitboard 4-in-a-row detection (horizontal, vertical, 2 diagonals)."""
    m = bb & (bb >> 7)          # horizontal
    if m & (m >> 14):
        return True
    m = bb & (bb >> 1)          # vertical
    if m & (m >> 2):
        return True
    m = bb & (bb >> 8)          # diagonal backslash
    if m & (m >> 16):
        return True
    m = bb & (bb >> 6)          # diagonal slash
    if m & (m >> 12):
        return True
    return False


def _flip_horizontal_u64(b: int) -> int:
    """Reverse column order of a Connect4 bitboard."""
    col0 = (b & 0x7F) << 42
    col1 = (b & (0x7F << 7)) << 28
    col2 = (b & (0x7F << 14)) << 14
    col3 = b & (0x7F << 21)
    col4 = (b & (0x7F << 28)) >> 14
    col5 = (b & (0x7F << 35)) >> 28
    col6 = (b & (0x7F << 42)) >> 42
    return col0 | col1 | col2 | col3 | col4 | col5 | col6


def _canonical_board(own: int, opp: int) -> tuple[int, int, bool]:
    """Return (canonical_own, canonical_opp, is_flipped)."""
    f_own = _flip_horizontal_u64(own)
    f_opp = _flip_horizontal_u64(opp)
    if (own, opp) <= (f_own, f_opp):
        return own, opp, False
    return f_own, f_opp, True


def _legal_mask(own: int, opp: int) -> int:
    """Return 7-bit mask of legal columns (bit c set if column c is playable)."""
    occupied = own | opp
    mask = 0
    for c in range(7):
        if (occupied & _COL_FULL[c]) != _COL_FULL[c]:
            mask |= 1 << c
    return mask


def _make_move(own: int, opp: int, col: int) -> tuple[int, int, str]:
    """Drop a piece in `col`. Returns (new_own, new_opp, result).

    After the move, own/opp are swapped (next player's perspective).
    `result` is one of 'win', 'draw', 'continue', 'illegal'.
    """
    col_shift = col * 7
    occupied = own | opp
    for r in range(ROWS):
        bit = 1 << (col_shift + r)
        if (occupied & bit) == 0:
            new_board = own | bit
            if _check_win(new_board):
                return opp, new_board, "win"
            if bin(new_board | opp).count("1") == 42:
                return opp, new_board, "draw"
            return opp, new_board, "continue"
    return own, opp, "illegal"


def _board_to_planes(own: int, opp: int) -> np.ndarray:
    """Convert bitboard to (3, 6, 7) float32: own, opp, turn-indicator."""
    planes = np.empty((3, ROWS, COLS), dtype=np.float32)
    planes[0] = (np.bitwise_and(_BIT_TABLE, np.uint64(own)) != np.uint64(0))
    planes[1] = (np.bitwise_and(_BIT_TABLE, np.uint64(opp)) != np.uint64(0))
    planes[2] = 1.0
    return planes


def _sample_action(policy: np.ndarray) -> int:
    """Sample one action from the policy distribution."""
    r = random.random()
    cumsum = 0.0
    for c in range(7):
        cumsum += policy[c]
        if r < cumsum:
            return c
    for c in range(6, -1, -1):
        if policy[c] > 0:
            return c
    return 0


# ─── MCTS Node ─────────────────────────────────────────────────────────

class _Node:
    """One node in the MCTS tree. Uses __slots__ for low overhead."""

    __slots__ = ("own", "opp", "children", "n", "w", "p", "expanded", "terminal")

    def __init__(self, own: int, opp: int) -> None:
        self.own = own
        self.opp = opp
        self.children: list[Optional[_Node]] = [None] * 7
        self.n = [0] * 7            # visit counts per action
        self.w = [0.0] * 7          # sum of backed-up values per action
        self.p = [0.0] * 7          # prior probabilities per action
        self.expanded = False
        self.terminal: Optional[float] = None  # None if non-terminal


# ─── Config ─────────────────────────────────────────────────────────────

@dataclass
class SelfPlayConfig:
    """Configuration for in-process self-play."""

    sims: int = 200                    # MCTS simulations per move
    batch_games: int = 32              # games played simultaneously
    c_puct: float = 1.5                # PUCT exploration constant
    temperature: float = 1.0           # visit-count temperature
    dirichlet_alpha: float = 0.3       # Dir. noise concentration
    dirichlet_epsilon: float = 0.25    # Dir. noise mixing weight
    buffer_size: int = 100_000         # max replay buffer samples
    weight_sync_games: int = 32        # sync model every N completed games


# ─── Replay Buffer ──────────────────────────────────────────────────────

class ReplayBuffer:
    """Thread-safe circular buffer for (planes, policy, value) samples."""

    def __init__(self, max_size: int = 100_000) -> None:
        self._lock = threading.Lock()
        self._data: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=max_size)
        self.total_pushed = 0

    def push(
        self,
        planes_list: list[np.ndarray],
        policy_list: list[np.ndarray],
        value_list: list[float],
    ) -> None:
        with self._lock:
            for pl, pi, v in zip(planes_list, policy_list, value_list):
                self._data.append((pl, pi, v))
            self.total_pushed += len(planes_list)

    def snapshot(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Thread-safe snapshot of the current buffer as numpy arrays."""
        with self._lock:
            n = len(self._data)
            if n == 0:
                return None
            items = list(self._data)
        planes = np.array([x[0] for x in items])
        policies = np.array([x[1] for x in items])
        values = np.array([x[2] for x in items], dtype=np.float32)
        return planes, policies, values

    def __len__(self) -> int:
        return len(self._data)


# ─── Self-Play Worker ───────────────────────────────────────────────────

class SelfPlayWorker:
    """Background thread that continuously generates self-play data.

    Uses a deep copy of the training model for inference. Weights are
    periodically synced from the training model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        buffer: ReplayBuffer,
        device: str,
        config: SelfPlayConfig,
    ) -> None:
        self.training_model = model
        self.sp_model = copy.deepcopy(
            model._orig_mod if hasattr(model, "_orig_mod") else model
        )
        self.sp_model.eval()
        self.buffer = buffer
        self.device = device
        self.config = config
        self.stop_event = threading.Event()
        self.games_played = 0
        self.samples_generated = 0
        self._thread: Optional[threading.Thread] = None
        self.eval_cache: dict[tuple[int, int], tuple[np.ndarray, float]] = {}

    def sync_weights(self) -> None:
        """Copy latest training model weights to the self-play model."""
        inner = (
            self.training_model._orig_mod
            if hasattr(self.training_model, "_orig_mod")
            else self.training_model
        )
        self.sp_model.load_state_dict(inner.state_dict())
        self.eval_cache.clear()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="selfplay")
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _evaluate_batch_cached(
        self, own_opp_list: list[tuple[int, int]]
    ) -> list[tuple[np.ndarray, float]]:
        """Evaluate a batch of (own, opp) boards using eval_cache and symmetry."""
        results: list[Optional[tuple[np.ndarray, float]]] = [None] * len(own_opp_list)
        missing_indices: list[int] = []
        missing_boards: list[tuple[int, int]] = []

        for i, (own, opp) in enumerate(own_opp_list):
            c_own, c_opp, is_flipped = _canonical_board(own, opp)
            cached = self.eval_cache.get((c_own, c_opp))
            if cached is not None:
                policy, val = cached
                if is_flipped:
                    policy = policy[::-1]
                results[i] = (policy, val)
            else:
                missing_indices.append(i)
                missing_boards.append((own, opp))

        if missing_boards:
            batch_planes = np.stack(
                [_board_to_planes(o, p) for o, p in missing_boards]
            )
            bt = torch.from_numpy(batch_planes).to(self.device)
            lp, vs, _moves_left, _confidence = self.sp_model(bt)
            probs = lp.exp().cpu().numpy()
            vals = vs.cpu().numpy().flatten()

            if len(self.eval_cache) > 500_000:
                self.eval_cache.clear()

            for idx, (own, opp), policy, val in zip(
                missing_indices, missing_boards, probs, vals
            ):
                c_own, c_opp, is_flipped = _canonical_board(own, opp)
                stored_policy = policy[::-1] if is_flipped else policy
                self.eval_cache[(c_own, c_opp)] = (stored_policy, float(val))
                results[idx] = (policy, float(val))

        return [r for r in results if r is not None]

    # ── Main loop ──────────────────────────────────────────────────────

    def _run(self) -> None:
        cfg = self.config
        n = cfg.batch_games

        # Per-game state: bitboard pairs + sample accumulators.
        owns = [0] * n
        opps = [0] * n
        game_planes: list[list[np.ndarray]] = [[] for _ in range(n)]
        game_policies: list[list[np.ndarray]] = [[] for _ in range(n)]

        with torch.no_grad():
            while not self.stop_event.is_set():
                # Build MCTS root for every active game.
                roots = [_Node(owns[i], opps[i]) for i in range(n)]

                # Batch-evaluate all roots with cache.
                root_evals = self._evaluate_batch_cached(
                    [(owns[i], opps[i]) for i in range(n)]
                )
                for i in range(n):
                    self._expand_root(roots[i], root_evals[i][0])

                # ── Simulation loop ────────────────────────────────────
                for _ in range(cfg.sims):
                    needs_eval: list[tuple[int, _Node, list]] = []

                    for i in range(n):
                        path, leaf, term_val = self._select(roots[i])
                        if term_val is not None:
                            self._backup(path, term_val)
                        else:
                            needs_eval.append((i, leaf, path))

                    if needs_eval:
                        evals = self._evaluate_batch_cached(
                            [(lf.own, lf.opp) for _, lf, _ in needs_eval]
                        )
                        for (idx, leaf, path), (policy, value) in zip(
                            needs_eval, evals
                        ):
                            self._expand_leaf(leaf, policy)
                            self._backup(path, value)

                # ── Make moves & collect finished games ─────────────────
                for i in range(n):
                    policy = self._get_policy(roots[i])
                    game_planes[i].append(_board_to_planes(owns[i], opps[i]))
                    game_policies[i].append(policy)

                    action = _sample_action(policy)
                    new_own, new_opp, result = _make_move(owns[i], opps[i], action)
                    owns[i], opps[i] = new_own, new_opp

                    if result in ("win", "draw"):
                        n_moves = len(game_planes[i])
                        values: list[float] = []
                        for j in range(n_moves):
                            if result == "draw":
                                values.append(0.0)
                            else:
                                values.append(
                                    1.0 if (n_moves - 1 - j) % 2 == 0 else -1.0
                                )

                        self.buffer.push(game_planes[i], game_policies[i], values)
                        self.games_played += 1
                        self.samples_generated += n_moves

                        # Reset slot for a new game.
                        owns[i], opps[i] = 0, 0
                        game_planes[i] = []
                        game_policies[i] = []

                        # Periodic weight sync.
                        if self.games_played % cfg.weight_sync_games == 0:
                            self.sync_weights()

    # ── MCTS helpers ───────────────────────────────────────────────────

    def _expand_root(self, node: _Node, policy: np.ndarray) -> None:
        """Expand root with NN priors + Dirichlet noise."""
        legal = _legal_mask(node.own, node.opp)
        legal_cols = [c for c in range(7) if legal & (1 << c)]
        n_legal = len(legal_cols)

        # Mask & normalize NN policy.
        masked = np.zeros(7, dtype=np.float32)
        for c in legal_cols:
            masked[c] = max(float(policy[c]), 1e-8)
        s = masked.sum()
        if s > 0:
            masked /= s

        # Mix in Dirichlet noise.
        eps = self.config.dirichlet_epsilon
        if eps > 0 and n_legal > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * n_legal)
            for idx, c in enumerate(legal_cols):
                node.p[c] = float((1 - eps) * masked[c] + eps * noise[idx])
        else:
            for c in range(7):
                node.p[c] = float(masked[c])

        node.expanded = True

    def _expand_leaf(self, node: _Node, policy: np.ndarray) -> None:
        """Expand a non-root leaf with NN priors (no noise)."""
        legal = _legal_mask(node.own, node.opp)
        for c in range(7):
            if legal & (1 << c):
                node.p[c] = max(float(policy[c]), 1e-8)
            else:
                node.p[c] = 0.0
        s = sum(node.p)
        if s > 0:
            inv_s = 1.0 / s
            for c in range(7):
                node.p[c] *= inv_s
        node.expanded = True

    def _select(self, root: _Node) -> tuple[list, _Node, Optional[float]]:
        """Walk from root to a leaf using PUCT. Returns (path, leaf, terminal_value)."""
        node = root
        path: list[tuple[_Node, int]] = []

        while True:
            if node.terminal is not None:
                return path, node, node.terminal

            if not node.expanded:
                return path, node, None

            action = self._puct_select(node)
            path.append((node, action))

            if node.children[action] is None:
                new_own, new_opp, result = _make_move(node.own, node.opp, action)
                child = _Node(new_own, new_opp)
                node.children[action] = child

                if result == "win":
                    child.terminal = -1.0  # from child's perspective: they lost
                    child.expanded = True
                    return path, child, -1.0
                elif result == "draw":
                    child.terminal = 0.0
                    child.expanded = True
                    return path, child, 0.0

            node = node.children[action]

    def _puct_select(self, node: _Node) -> int:
        """Select action with highest PUCT score."""
        c_puct = self.config.c_puct
        total_n = sum(node.n)
        sqrt_total = math.sqrt(total_n) if total_n > 0 else 0.0

        best_score = -1e9
        best_action = 0
        for c in range(7):
            if node.p[c] == 0.0:
                continue
            q = node.w[c] / node.n[c] if node.n[c] > 0 else 0.0
            u = c_puct * node.p[c] * sqrt_total / (1 + node.n[c])
            score = q + u
            if score > best_score:
                best_score = score
                best_action = c
        return best_action

    def _backup(self, path: list[tuple[_Node, int]], value: float) -> None:
        """Back up value through the path, flipping sign at each level."""
        v = value
        for node, action in reversed(path):
            v = -v
            node.w[action] += v
            node.n[action] += 1

    def _get_policy(self, root: _Node) -> np.ndarray:
        """Extract visit-count policy from root."""
        visits = np.array(root.n, dtype=np.float32)
        temp = self.config.temperature

        if temp < 0.01:
            policy = np.zeros(7, dtype=np.float32)
            if visits.max() > 0:
                policy[np.argmax(visits)] = 1.0
            return policy

        powered = visits ** (1.0 / temp)
        total = powered.sum()
        if total > 0:
            return powered / total
        return np.full(7, 1.0 / 7, dtype=np.float32)
