"""
gui.py — Play Connect 4 against Connect4Net in a Pygame window.

The human plays "Red" (drops first). The AI plays "Yellow" using the
trained model's policy head argmax (with a legality mask). The value
head is shown as a small bar at the top of the window.

Controls
--------
* Left-click on a column to drop a Red piece in that column.
* Press R to reset the game.
* Press Q or close the window to quit.

Display
-------
* Blue board with white grid lines.
* Pieces animate: they fall from the top of the column to the lowest
  empty cell over a few frames. The animation is purely visual — the
  game state advances instantly on click.

A note on strength
------------------
The AI is only as good as the model that's been trained. After 0 cycles
of self-play the model is random; after dozens of cycles (and depending
on sim count) it should be a competent amateur. The first usable signal
usually appears around 1k–2k training samples (a few self-play games).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pygame
import torch

from model import Connect4Net


# --- Visual constants ------------------------------------------------------
WINDOW_W = 700
WINDOW_H = 720            # extra strip at the top for the value bar
BOARD_TOP = 60
CELL_W = (WINDOW_W - 40) // 7
CELL_H = (WINDOW_H - BOARD_TOP - 40) // 6
PIECE_R = min(CELL_W, CELL_H) // 2 - 4
COLORS = {
    "bg":       (24, 24, 36),
    "board":    (40, 80, 200),
    "hole":     (24, 24, 36),
    "red":      (220, 50, 50),
    "yellow":   (240, 210, 50),
    "grid":     (200, 200, 240),
    "text":     (240, 240, 240),
    "value_pos":(80, 200, 80),
    "value_neg":(200, 80, 80),
    "value_zero":(150, 150, 150),
}
FPS = 60
ANIM_FRAMES = 12  # frames per falling-piece animation


# --- Bitboard helpers (mirror of the Rust side) ----------------------------

def col_mask(c: int) -> int:
    return 0x7F << (c * 7)


def set_bit(bitboard: int, r: int, c: int) -> int:
    return bitboard | (1 << (c * 7 + r))


def find_row(bitboard: int, c: int) -> int | None:
    """Return the lowest empty row in column `c`, or None if full."""
    occ = bitboard & col_mask(c)
    if occ == col_mask(c):
        return None
    # Carry trick (same as Rust): with the top guard set, adding the bottom
    # bit rolls a carry through occupied cells and lands on the first empty.
    bottom = 1 << (c * 7)
    next_pos = ((occ | (1 << (c * 7 + 6))) + bottom) & col_mask(c)
    bit = next_pos.bit_length() - 1
    return bit - c * 7  # row index within the column


def has_win(bitboard: int) -> bool:
    """Same 4-shift trick as Rust `check_win`."""
    SHIFT = 1
    # Mask out the guard bits at the top of each column.
    b = bitboard & ~0x0001_0204_0810_2040
    # Vertical (within column)
    if (b & (b >> 1) & (b >> 2) & (b >> 3)) != 0:
        return True
    # Horizontal (across columns)
    if (b & (b >> 7) & (b >> 14) & (b >> 21)) != 0:
        return True
    # Diagonal \
    if (b & (b >> 6) & (b >> 12) & (b >> 18)) != 0:
        return True
    # Diagonal /
    if (b & (b >> 8) & (b >> 16) & (b >> 24)) != 0:
        return True
    return False


# --- Network inference ----------------------------------------------------

def board_to_planes(red: int, yellow: int) -> np.ndarray:
    """Canonical: own = current player to move, opp = the other."""
    planes = np.zeros((3, 6, 7), dtype=np.float32)
    for r in range(6):
        for c in range(7):
            bit = 1 << (c * 7 + r)
            if red & bit:
                planes[0, r, c] = 1.0  # own
            elif yellow & bit:
                planes[1, r, c] = 1.0  # opp
            else:
                planes[2, r, c] = 1.0
    return planes


def ai_select(model: Connect4Net, own: int, opp: int, device: torch.device) -> int:
    """Pick the column with the highest policy probability, masking illegal moves."""
    planes = board_to_planes(own, opp)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    with torch.no_grad():
        log_p, v = model(x)
    p = log_p.exp().cpu().numpy()[0]
    # Mask illegal columns.
    occ = own | opp
    for c in range(7):
        if (occ & col_mask(c)) == col_mask(c):
            p[c] = 0.0
    return int(np.argmax(p))


# --- Game state -----------------------------------------------------------

class Game:
    def __init__(self) -> None:
        # Bitboards: red plays first.
        self.red = 0
        self.yellow = 0
        # Track whose turn it is: 0 = red (human), 1 = yellow (AI).
        self.turn = 0
        self.winner: int | None = None  # 0=red, 1=yellow, 2=draw, None=ongoing
        # For animation: when a piece starts falling, store (col, target_row, color, frame).
        self.anim: tuple | None = None

    def legal_columns(self) -> list[int]:
        occ = self.red | self.yellow
        return [c for c in range(7) if (occ & col_mask(c)) != col_mask(c)]

    def is_full(self) -> bool:
        return (self.red | self.yellow).bit_count() == 42

    def drop(self, c: int) -> int | None:
        """Drop a piece in column `c` for the current player. Returns the
        row index of the new piece, or None if the column is full."""
        bb = self.red if self.turn == 0 else self.yellow
        row = find_row(bb, c)
        if row is None:
            return None
        if self.turn == 0:
            self.red = set_bit(self.red, row, c)
        else:
            self.yellow = set_bit(self.yellow, row, c)
        # Check for win. The bitboard before any swap is what was just placed.
        if self.turn == 0 and has_win(self.red):
            self.winner = 0
        elif self.turn == 1 and has_win(self.yellow):
            self.winner = 1
        elif self.is_full():
            self.winner = 2
        # Switch turn.
        self.turn = 1 - self.turn
        return row


# --- Drawing --------------------------------------------------------------

def draw_board(screen: pygame.Surface, game: Game) -> None:
    screen.fill(COLORS["bg"])
    # Value bar at the top.
    bar_h = 24
    pygame.draw.rect(screen, COLORS["board"], (0, 0, WINDOW_W, bar_h))
    label = pygame.font.SysFont("consolas", 18).render(
        f"Turn: {'RED (you)' if game.turn == 0 else 'YELLOW (AI)'}",
        True, COLORS["text"]
    )
    screen.blit(label, (10, 3))
    if game.winner is None:
        status = ""
    elif game.winner == 2:
        status = "  —  DRAW"
    else:
        status = f"  —  {'RED' if game.winner == 0 else 'YELLOW'} WINS"
    s = pygame.font.SysFont("consolas", 18).render(status, True, COLORS["text"])
    screen.blit(s, (WINDOW_W // 2, 3))

    # Board background.
    board_rect = pygame.Rect(
        20, BOARD_TOP,
        CELL_W * 7, CELL_H * 6
    )
    pygame.draw.rect(screen, COLORS["board"], board_rect)

    # Pieces.
    for r in range(6):
        for c in range(7):
            cx = 20 + c * CELL_W + CELL_W // 2
            cy = BOARD_TOP + (5 - r) * CELL_H + CELL_H // 2  # row 0 at the bottom
            bit = 1 << (c * 7 + r)
            if game.red & bit:
                color = COLORS["red"]
            elif game.yellow & bit:
                color = COLORS["yellow"]
            else:
                color = COLORS["hole"]
            pygame.draw.circle(screen, color, (cx, cy), PIECE_R)

    # Animation overlay: a single falling piece.
    if game.anim is not None:
        col, target_row, color, frame = game.anim
        cx = 20 + col * CELL_W + CELL_W // 2
        # Interpolate y from above the board to the target cell.
        target_y = BOARD_TOP + (5 - target_row) * CELL_H + CELL_H // 2
        start_y = BOARD_TOP - CELL_H
        t = frame / ANIM_FRAMES
        # Ease-out for a nicer feel.
        t = 1 - (1 - t) ** 2
        cy = int(start_y + t * (target_y - start_y))
        pygame.draw.circle(screen, color, (cx, cy), PIECE_R)

    pygame.display.flip()


# --- Main loop ------------------------------------------------------------

def run(model_path: str | None, device_str: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Connect 4 — vs Connect4Net")
    clock = pygame.time.Clock()

    # Load the model.
    device = torch.device(device_str)
    net = Connect4Net().to(device).eval()
    if model_path and os.path.exists(model_path):
        net.load(model_path, map_location=device)
        print(f"[gui] loaded model from {model_path}")
    else:
        print(f"[gui] no model found at {model_path!r} — using random weights")

    game = Game()
    ai_thinking = False
    ai_think_start = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    game = Game()
                    ai_thinking = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game.turn == 0 and game.winner is None and game.anim is None and not ai_thinking:
                    mx = event.pos[0]
                    col = max(0, min(6, (mx - 20) // CELL_W))
                    row = game.drop(col)
                    if row is not None:
                        game.anim = (col, row, COLORS["red"], 0)

        # Animate falling piece.
        if game.anim is not None:
            col, target_row, color, frame = game.anim
            frame += 1
            if frame >= ANIM_FRAMES:
                game.anim = None
                # After Red's animation, kick off the AI.
                if game.winner is None and game.turn == 1:
                    ai_thinking = True
                    ai_think_start = time.time()
            else:
                game.anim = (col, target_row, color, frame)

        # AI turn.
        if ai_thinking and game.anim is None:
            # Always show the "thinking" state for at least one frame.
            if time.time() - ai_think_start > 0.15:
                with torch.no_grad():
                    col = ai_select(net, game.yellow, game.red, device)
                if find_row(game.yellow, col) is not None:
                    game.turn = 1
                    row = game.drop(col)
                    if row is not None:
                        game.anim = (col, row, COLORS["yellow"], 0)
                ai_thinking = False

        draw_board(screen, game)
        clock.tick(FPS)

    pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser(description="Play Connect 4 vs Connect4Net")
    p.add_argument("--model", default="connect4_model.pt", help="path to .pt state_dict")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    try:
        run(args.model, args.device)
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
