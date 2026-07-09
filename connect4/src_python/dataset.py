"""
dataset.py — Read the C4D1 self-play binary file into PyTorch tensors.

Binary format (C4D1)
-------------------
Header (16 bytes):
    4 bytes  : magic = b"C4D1"             (0x43 0x34 0x44 0x31)
    4 bytes  : u32 LE  sample count N
    8 bytes  : reserved, zero

Sample (56 bytes), N times:
    8 bytes  : u64 LE  own bitboard       (current player's pieces)
    8 bytes  : u64 LE  opponent bitboard  (the other player's pieces)
    8 bytes  : u64 LE  turn mask          (all 1s — constant bias)
    28 bytes : 7 × f32 LE  MCTS policy pi
    4 bytes  : f32 LE    game outcome z ∈ {-1, 0, +1}

The bitboard layout is column-major with a guard bit at the top of each
column — see `bitboard.rs` in the Rust side for the full encoding. The
unpacking into a 3×6×7 tensor here drops the guard bits (they only exist
in the Rust representation to make the 4-shift win check safe).

Bit-to-grid mapping
-------------------
For row r ∈ {0..=5} and column c ∈ {0..=6}, the cell maps to bit index
`c * 7 + r` in the u64. Row 0 is the bottom, row 5 is the top. We unpack
each u64 into a (3, 6, 7) array of float32s — three planes are the own /
opponent / "empty-or-guard" mask. Plane 2 is redundant given the other
two but is the format contract for compatibility with future net
architectures that want a turn/empty plane.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# Constants — keep in sync with the Rust side.
MAGIC = b"C4D1"
HEADER_SIZE = 16
SAMPLE_SIZE = 56
N_PLANES = 3
BOARD_H = 6
BOARD_W = 7
N_COLS = 7
POLICY_SIZE = N_COLS


def decode_bitboard(own: int, opp: int) -> np.ndarray:
    """Decode the (own, opp) bitboard pair into a (3, 6, 7) float32 array.

    Plane layout:
        0 : own pieces    (1.0 where the current player has a piece)
        1 : opponent      (1.0 where the opponent has a piece)
        2 : empty + turn  (1.0 elsewhere; the third u64 in the format
                           is all 1s, so this plane is just a bias)

    For each cell (r, c), the corresponding bit in the u64 is at index
    `c * 7 + r`. The guard bit at index `c * 7 + 6` is dropped here
    (it's never set in valid states, but defensively excluded).
    """
    planes = np.zeros((N_PLANES, BOARD_H, BOARD_W), dtype=np.float32)

    # Fast path: precompute the bit mask for each cell. We can vectorize
    # the per-sample decode by lifting (own, opp) to numpy arrays and
    # using bitwise operations on the (N, 6, 7) view of the bits. For a
    # single sample the Python loop is fine — 42 iterations is nothing.
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            bit = 1 << (c * 7 + r)
            if own & bit:
                planes[0, r, c] = 1.0
            elif opp & bit:
                planes[1, r, c] = 1.0
            else:
                planes[2, r, c] = 1.0
    return planes


def decode_bitboard_batched(
    own_arr: np.ndarray, opp_arr: np.ndarray
) -> np.ndarray:
    """Vectorized batched decoder for `N` samples.

    Args:
        own_arr: (N,) uint64 numpy array of own bitboards.
        opp_arr: (N,) uint64 numpy array of opponent bitboards.

    Returns:
        (N, 3, 6, 7) float32 numpy array of input planes.
    """
    N = own_arr.shape[0]
    planes = np.zeros((N, N_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
    # Precompute the (6, 7) bit-index table.
    bit_idx = np.arange(BOARD_H * BOARD_W).reshape(BOARD_W, BOARD_H).T  # (6, 7)
    bit_idx = (bit_idx * 0 + np.arange(BOARD_W)[None, :]) * 7 + np.arange(BOARD_H)[:, None]
    # ^ Equivalent to: bit_idx[r, c] = c * 7 + r.

    for r in range(BOARD_H):
        for c in range(BOARD_W):
            bit = np.uint64(1) << np.uint64(c * 7 + r)
            own_mask = (own_arr & bit) != 0
            opp_mask = (opp_arr & bit) != 0
            empty_mask = ~(own_mask | opp_mask)
            planes[own_mask, 0, r, c] = 1.0
            planes[opp_mask, 1, r, c] = 1.0
            planes[empty_mask, 2, r, c] = 1.0
    return planes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class C4Dataset(Dataset):
    """Memory-mapped reader for the C4D1 self-play binary file.

    The full file is loaded into a single uint8 numpy array of shape
    (16 + 56 * N,). For a typical 64-game run this is ~1.5 MB — small
    enough to fit comfortably in RAM but big enough that we'd rather
    not reopen the file on every `__getitem__`. The dataset slices
    into the in-memory buffer.

    For multi-million-sample runs you may want to mmap the file instead
    of loading it. The interface would be identical; only the underlying
    storage changes.
    """

    def __init__(self, path: str, max_samples: int | None = None) -> None:
        self.path = path
        with open(path, "rb") as f:
            header = f.read(HEADER_SIZE)
        if header[:4] != MAGIC:
            raise ValueError(
                f"{path}: bad magic {header[:4]!r}, expected {MAGIC!r} "
                f"(was this file produced by `connect4-selfplay`?)"
            )
        declared = int.from_bytes(header[4:8], "little")
        # Sanity-check declared size matches file length.
        import os
        actual = (os.path.getsize(path) - HEADER_SIZE) // SAMPLE_SIZE
        if declared != actual:
            raise ValueError(
                f"{path}: header says {declared} samples but file has {actual}"
            )
        self.count = min(declared, max_samples) if max_samples else declared

        # Load the entire data section (skip the header) as a uint8 array.
        # np.fromfile is fast for contiguous binary data.
        self._raw = np.fromfile(
            path, dtype=np.uint8, count=HEADER_SIZE + SAMPLE_SIZE * self.count
        )[HEADER_SIZE:]

        # Pre-decode all planes into one big array — way faster than
        # calling decode_bitboard on every __getitem__.
        # Shape after reshape: (count, 56) bytes, then split:
        #   own:   bytes 0..8   (uint64)
        #   opp:   bytes 8..16  (uint64)
        #   _turn: bytes 16..24 (unused)
        #   pi:    bytes 24..52 (7 × f32)
        #   z:     bytes 52..56 (1 × f32)
        reshaped = self._raw.reshape(self.count, SAMPLE_SIZE)
        self._own = np.frombuffer(reshaped[:, 0:8].tobytes(), dtype=np.uint64).copy()
        self._opp = np.frombuffer(reshaped[:, 8:16].tobytes(), dtype=np.uint64).copy()
        self._policy = np.frombuffer(reshaped[:, 24:52].tobytes(), dtype=np.float32).reshape(self.count, POLICY_SIZE).copy()
        self._value = np.frombuffer(reshaped[:, 52:56].tobytes(), dtype=np.float32).copy()

        # Pre-decode planes as (N, 3, 6, 7) float32. ~3 MB for 64 games.
        self._planes = decode_bitboard_batched(self._own, self._opp)

        # Symmetry augmentation: Connect 4 is invariant under horizontal flip
        # (col c <-> col 6-c). When `symmetry=True`, each __getitem__ randomly
        # returns the original or the flipped version (50/50). Effectively
        # doubles the dataset with zero extra self-play cost.
        self.symmetry = False

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planes = self._planes[idx]      # (3, 6, 7) numpy view
        policy = self._policy[idx]      # (7,)
        value = self._value[idx]        # ()

        if self.symmetry:
            # 50/50: with prob 0.5 apply horizontal flip.
            # No RNG of our own — torch/numpy ops use the default global RNG,
            # which is fine for data aug (the model doesn't need reproducible batches).
            import random as _r
            if _r.random() < 0.5:
                planes = planes[:, :, ::-1].copy()  # flip cols (axis=2)
                policy = policy[::-1].copy()        # flip policy columns

        return (
            torch.from_numpy(planes),
            torch.from_numpy(policy),
            torch.tensor(value, dtype=torch.float32),
        )

    def stats(self) -> dict:
        """Summary statistics for the dataset — useful in train.py for logging."""
        n = self.count
        n_pos = int((self._value > 0).sum())
        n_neg = int((self._value < 0).sum())
        n_draw = int((self._value == 0).sum())
        avg_plies = n / max(1, n)  # we don't track game boundaries here
        return {
            "samples": n,
            "wins": n_pos,
            "losses": n_neg,
            "draws": n_draw,
            "win_rate": n_pos / max(1, n_pos + n_neg),
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python dataset.py <path-to-selfplay.bin>")
        sys.exit(2)
    ds = C4Dataset(sys.argv[1])
    print(f"loaded {len(ds):,} samples from {sys.argv[1]}")
    print(f"stats: {ds.stats()}")
    planes, policy, value = ds[0]
    print(f"sample 0: planes {tuple(planes.shape)}  policy {policy.tolist()}  value {value.item():+.0f}")
    print(f"planes[0] (own):\n{planes[0].numpy()}")
    print(f"planes[1] (opp):\n{planes[1].numpy()}")

    # Symmetry smoke check: flip a sample and verify the bit positions
    # move correctly (col c -> col 6-c).
    import numpy as _np
    own = (1 << 0) | (1 << 8) | (1 << 21)        # (r=0,c=0), (r=1,c=1), (r=3,c=3)
    opp = (1 << 14) | (1 << 28)                  # (r=2,c=2), (r=4,c=4)
    test_planes = decode_bitboard(own, opp)
    flipped = test_planes[:, :, ::-1].copy()
    assert flipped[0, 0, 0] == 0.0, "(0,0) was empty, still empty after flip"
    assert flipped[0, 0, 6] == 1.0, "own at (0,0) should move to (0,6)"
    assert flipped[0, 3, 3] == 1.0, "own at (3,3) should stay at (3,3) (center)"
    print("dataset symmetry smoke check: OK")
