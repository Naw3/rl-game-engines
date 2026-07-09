// =============================================================================
// bitboard.rs — Pure-math referee for Connect 4.
//
// Encoding (column-major, 7 bits per column, 7 columns, 1 guard bit per column):
//
//   col 0: bits  0..=6   (data 0..=5, guard 6)
//   col 1: bits  7..=13
//   col 2: bits 14..=20
//   col 3: bits 21..=27
//   col 4: bits 28..=34
//   col 5: bits 35..=41
//   col 6: bits 42..=48
//
//   Total: 49 used bits in a u64 (bits 49..=63 are always 0).
//
// The "current player" perspective is canonical: `own` is always the side to
// move. After each `make_move`, we swap `own` and `opp` so the invariant
// holds for the next position. This means the network always sees the board
// from the current player's point of view, with no separate turn bit needed
// in the input planes (though we still pass one for symmetry with AlphaZero).
//
// Win detection uses the classic 4-shift trick:
//   vertical  : shift 1   (within a column, one row up)
//   horizontal: shift 7   (within a row, one column right)
//   diag \    : shift 6   (down-right when reading rows top-to-bottom)
//   diag /    : shift 8   (up-right when reading rows top-to-bottom)
//
// A single AND/AND/AND/AND non-zero check catches all four directions in O(1).
// The guard bit at the top of each column is essential: it would otherwise be
// possible to form a "horizontal" alignment out of pure guard bits {6, 13, 20,
// 27, ...} spaced exactly 7 apart. We mask guards out in `check_win` as a
// defensive belt-and-suspenders even though we never set them in valid boards.
// =============================================================================

/// Outcome of attempting a move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveResult {
    /// Move was legal, game continues.
    Continue,
    /// Move was legal and created a 4-in-a-row for the mover.
    Win,
    /// Move was legal, board is now full (42 pieces), no winner.
    Draw,
    /// The requested column was full or out of range.
    Illegal,
}

/// Board state, canonical to the player to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Board {
    pub own: u64,
    pub opp: u64,
}

/// Top-of-column guard bits. Set in `BOARD_MASK`, never set in a valid board.
const BOARD_MASK: u64 = 0x0001_0204_0810_2040;

/// Returns the column mask for column `c` (all 7 bits including guard).
#[inline]
pub const fn col_mask(c: usize) -> u64 {
    0x7Fu64 << (c * 7)
}

/// Returns the bottom-bit of column `c` (the row-0 bit).
#[inline]
pub const fn col_bottom(c: usize) -> u64 {
    1u64 << (c * 7)
}

/// Returns the guard bit of column `c` (row 6, sentinel).
#[inline]
pub const fn col_top_guard(c: usize) -> u64 {
    1u64 << (c * 7 + 6)
}

/// Returns the bit index of the lowest empty cell in column `c`, or `None`
/// if the column is full. Uses the standard "carry" trick: with the guard
/// bit pre-set, adding the bottom bit rolls a carry up through the occupied
/// cells and lands on the first empty one.
#[inline]
pub fn next_empty_bit(occupied: u64, c: usize) -> Option<u64> {
    let mask = col_mask(c);
    let occ = occupied & mask;
    if occ == mask {
        return None; // column is full
    }
    let next = ((occ | col_top_guard(c)) + col_bottom(c)) & mask;
    debug_assert_eq!(next.count_ones(), 1, "next_empty_bit must be a single bit");
    Some(next)
}

/// 4-in-a-row check for a single player's u64. O(1), no loops.
#[inline]
pub fn check_win(board: u64) -> bool {
    // Belt and suspenders: strip guards in case a future code path ever sets
    // them. Valid play never does.
    let b = board & !BOARD_MASK;

    // Vertical (within a column): 4 adjacent bits stacked up.
    let m = b & (b >> 1);
    if m & (b >> 2) & (b >> 3) != 0 {
        return true;
    }
    // Horizontal (across columns, one row): shift by 7.
    let m = b & (b >> 7);
    if m & (b >> 14) & (b >> 21) != 0 {
        return true;
    }
    // Diagonal \ (down-right): shift by 6 (one column right = +7, one row down = -1).
    let m = b & (b >> 6);
    if m & (b >> 12) & (b >> 18) != 0 {
        return true;
    }
    // Diagonal / (up-right): shift by 8 (one column right = +7, one row up = +1).
    let m = b & (b >> 8);
    if m & (b >> 16) & (b >> 24) != 0 {
        return true;
    }
    false
}

impl Board {
    /// Empty board. `own` is the side to move (convention: player 1 = "Red",
    /// but the bit encoding makes no distinction — only the perspective swap
    /// after each move matters).
    pub const fn new() -> Self {
        Board { own: 0, opp: 0 }
    }

    /// 7-bit bitmask: bit `c` is 1 iff column `c` is a legal move.
    #[inline]
    pub fn legal_moves(&self) -> u8 {
        let mut mask = 0u8;
        let occ = self.own | self.opp;
        for c in 0..7usize {
            if (occ & col_mask(c)) != col_mask(c) {
                mask |= 1 << c;
            }
        }
        mask
    }

    /// Number of pieces on the board.
    #[inline]
    pub fn ply(&self) -> u32 {
        (self.own | self.opp).count_ones()
    }

    /// Apply a move in column `c`. Returns the outcome. After a legal move
    /// (Continue, Win, or Draw), the perspective is swapped so `own` again
    /// refers to the side to move.
    pub fn make_move(&mut self, c: usize) -> MoveResult {
        if c >= 7 {
            return MoveResult::Illegal;
        }
        let occ = self.own | self.opp;
        let piece = match next_empty_bit(occ, c) {
            Some(p) => p,
            None => return MoveResult::Illegal,
        };

        // Drop the piece in `own` (the mover).
        self.own |= piece;

        // Check win for the mover before swapping.
        let result = if check_win(self.own) {
            MoveResult::Win
        } else if self.ply() == 42 {
            MoveResult::Draw
        } else {
            MoveResult::Continue
        };

        // Hand the turn over: previous own -> opp, previous opp -> own.
        std::mem::swap(&mut self.own, &mut self.opp);

        result
    }

    /// Pack the three input planes as u64s (one bit per cell, column-major).
    ///   plane 0 = own pieces
    ///   plane 1 = opponent pieces
    ///   plane 2 = turn indicator (all 1s; provides a constant bias to the
    ///             network and signals that the input is canonical to the
    ///             player to move)
    pub fn to_planes(&self) -> [u64; 3] {
        [self.own, self.opp, !0u64]
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Unit tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_no_win() {
        let b = Board::new();
        assert!(!check_win(b.own));
        assert_eq!(b.legal_moves(), 0b0111_1111);
    }

    #[test]
    fn vertical_win() {
        // 4 in column 3 (middle column), bottom to row 3.
        let mut b = Board::new();
        // X plays column 3 four times. Each make_move swaps perspective, so
        // X plays columns 3, 0, 3, 0, 3, 0, 3 (alternating filler column).
        for col in [3, 0, 3, 0, 3, 0, 3] {
            let r = b.make_move(col);
            // 4th X move should win.
            if col == 3 && b.ply() == 7 {
                // After 3 X moves in col 3 and 3 O moves in col 0,
                // the 4th X move in col 3 should win.
                // We check after the move.
            }
            if col == 3 {
                // On 4th X move in col 3, b.own (now containing what was
                // previously opp, i.e. O) doesn't have 4 in a row. Wait,
                // the win is detected BEFORE the swap, so the returned
                // result is what matters.
                let _ = r;
            }
        }
        // Reset and test more carefully.
        let mut b = Board::new();
        b.own = 0; // X = own
        b.opp = 0;
        // Manually place 4 X in col 3 rows 0..4 (bits 21, 22, 23, 24)
        b.own = (1 << 21) | (1 << 22) | (1 << 23) | (1 << 24);
        assert!(check_win(b.own));
    }

    #[test]
    fn horizontal_win() {
        // 4 in a row across columns 0..4 at row 2.
        // Row 2 cells: bits 2, 9, 16, 23, 30, 37, 44.
        let b = Board {
            own: (1 << 2) | (1 << 9) | (1 << 16) | (1 << 23),
            opp: 0,
        };
        assert!(check_win(b.own));
    }

    #[test]
    fn diagonal_win() {
        // \ diagonal: col 0 row 0, col 1 row 1, col 2 row 2, col 3 row 3.
        // bits: 0, 8, 16, 24.
        let b = Board {
            own: (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24),
            opp: 0,
        };
        assert!(check_win(b.own));
    }

    #[test]
    fn anti_diagonal_win() {
        // / diagonal: col 0 row 3, col 1 row 2, col 2 row 1, col 3 row 0.
        // bits: 3, 9, 15, 21.
        let b = Board {
            own: (1 << 3) | (1 << 9) | (1 << 15) | (1 << 21),
            opp: 0,
        };
        assert!(check_win(b.own));
    }

    #[test]
    fn no_false_positive_from_guard_bits() {
        // Hypothetically if all 7 guard bits were set, they would form a
        // "horizontal" alignment {6, 13, 20, 27}. The BOARD_MASK strip in
        // check_win prevents that.
        let b = BOARD_MASK;
        assert!(!check_win(b));
    }

    #[test]
    fn make_move_swaps_perspective() {
        let mut b = Board::new();
        b.make_move(3);
        // After X plays col 3, perspective swaps. The bit should now be in
        // `opp` (it represents X from O's perspective).
        assert_eq!(b.own, 0);
        assert_ne!(b.opp, 0);
        assert_eq!(b.opp, 1u64 << (3 * 7));
    }

    #[test]
    fn full_column_is_illegal() {
        let mut b = Board::new();
        // Fill column 0 with 6 pieces (alternating X/O).
        for _ in 0..6 {
            let _ = b.make_move(0);
        }
        // 7th move in col 0 should be illegal.
        assert_eq!(b.make_move(0), MoveResult::Illegal);
    }

    #[test]
    fn carry_trick_finds_next_empty() {
        // Empty column 2 (bits 14..20).
        assert_eq!(next_empty_bit(0, 2), Some(1 << 14));
        // 3 pieces in col 2: rows 0, 1, 2 (bits 14, 15, 16).
        let occ = (1 << 14) | (1 << 15) | (1 << 16);
        assert_eq!(next_empty_bit(occ, 2), Some(1 << 17));
        // 6 pieces in col 2 (full): bits 14..19.
        let occ = (1 << 14) | (1 << 15) | (1 << 16) | (1 << 17) | (1 << 18) | (1 << 19);
        assert_eq!(next_empty_bit(occ, 2), None);
    }
}
