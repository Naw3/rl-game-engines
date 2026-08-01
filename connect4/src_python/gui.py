"""
gui.py — Play Connect 4 against Connect4Net in a Pygame window.

The human plays "Red" (drops first). The AI plays "Yellow" using the
batched MCTS agent from inference.py. The selected model is resolved from
best_model.json when the default ``auto`` path is used.

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
import sys
import time

import pygame


# --- Visual constants ------------------------------------------------------
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from config import CONFIG
    WINDOW_W = CONFIG.gui.window_w
    WINDOW_H = CONFIG.gui.window_h
    BOARD_TOP = CONFIG.gui.board_top
    FPS = CONFIG.gui.fps
    ANIM_FRAMES = CONFIG.gui.anim_frames
    COLORS = CONFIG.gui.colors
except Exception as err:
    print(f"[gui] WARNING: Failed to load config.py ({err}); using fallbacks")
    WINDOW_W = 700
    WINDOW_H = 720
    BOARD_TOP = 60
    FPS = 60
    ANIM_FRAMES = 12
    COLORS = {
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
    }

CELL_W = (WINDOW_W - 40) // 7
CELL_H = (WINDOW_H - BOARD_TOP - 40) // 6
PIECE_R = min(CELL_W, CELL_H) // 2 - 4


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
    bit = (next_pos & -next_pos).bit_length() - 1
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
        occ = self.red | self.yellow
        row = find_row(occ, c)
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

def draw_board(screen: pygame.Surface, game: Game, ai_thinking: bool = False, thinking_text: str = "") -> None:
    screen.fill(COLORS["bg"])

    # Header / Bar at the top.
    bar_h = 36
    pygame.draw.rect(screen, (30, 32, 48), (0, 0, WINDOW_W, bar_h))

    font_main = pygame.font.SysFont("segoeui", 18, bold=True)
    font_sub = pygame.font.SysFont("consolas", 16)

    if ai_thinking:
        turn_str = "AI is thinking..." if not thinking_text else f"AI is thinking... ({thinking_text})"
        turn_color = COLORS["yellow"]
    else:
        turn_str = "Your Turn (RED)" if game.turn == 0 else "AI Turn (YELLOW)"
        turn_color = COLORS["red"] if game.turn == 0 else COLORS["yellow"]

    label = font_main.render(turn_str, True, turn_color)
    screen.blit(label, (16, 6))

    if game.winner is not None:
        if game.winner == 2:
            status = "DRAW GAME!"
            status_color = (200, 200, 200)
        else:
            winner_name = "YOU WIN! (RED)" if game.winner == 0 else "AI WINS! (YELLOW)"
            status_color = COLORS["red"] if game.winner == 0 else COLORS["yellow"]
            status = winner_name

        s = font_main.render(status, True, status_color)
        screen.blit(s, (WINDOW_W // 2 - 60, 6))

        # Draw Play Again button
        btn_rect = pygame.Rect(WINDOW_W - 150, 4, 130, 28)
        pygame.draw.rect(screen, (60, 140, 230), btn_rect, border_radius=6)
        btn_s = font_sub.render("PLAY AGAIN", True, (255, 255, 255))
        screen.blit(btn_s, (WINDOW_W - 140, 8))

    # Board background with rounded corners
    board_rect = pygame.Rect(
        20, BOARD_TOP,
        CELL_W * 7, CELL_H * 6
    )
    pygame.draw.rect(screen, COLORS["board"], board_rect, border_radius=12)

    # Pieces.
    anim_col = -1
    anim_row = -1
    if game.anim is not None:
        anim_col, anim_row, _, _ = game.anim
        
    for r in range(6):
        for c in range(7):
            cx = 20 + c * CELL_W + CELL_W // 2
            cy = BOARD_TOP + (5 - r) * CELL_H + CELL_H // 2  # row 0 at the bottom
            bit = 1 << (c * 7 + r)
            
            if c == anim_col and r == anim_row:
                color = COLORS["hole"]
            elif game.red & bit:
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
        target_y = BOARD_TOP + (5 - target_row) * CELL_H + CELL_H // 2
        start_y = BOARD_TOP - CELL_H
        t = frame / ANIM_FRAMES
        t = 1 - (1 - t) ** 2  # Ease-out
        cy = int(start_y + t * (target_y - start_y))
        pygame.draw.circle(screen, color, (cx, cy), PIECE_R)

    pygame.display.flip()


# --- Main loop ------------------------------------------------------------

from src_python.inference import Connect4Agent, format_duration


def run(model_path: str | None, device_str: str, backend: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Connect 4 — vs Connect4Net (MCTS Agent)")
    clock = pygame.time.Clock()

    # Load search parameters from InferConfig
    max_think_time = getattr(CONFIG.infer, "max_think_time", 1.0)
    agent = Connect4Agent(
        model_path or _DEFAULT_GUI_MODEL,
        device=device_str,
        backend=backend,
    )
    stop_mode = "confidence stop" if agent.has_confidence_head else "time fallback"

    game = Game()
    ai_thinking = False
    ai_think_start = 0.0
    thinking_info = ""

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game = Game()
                ai_thinking = False
                thinking_info = ""
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                
                # Check Play Again button click
                if game.winner is not None:
                    if WINDOW_W - 150 <= mx <= WINDOW_W - 20 and 4 <= my <= 32:
                        game = Game()
                        ai_thinking = False
                        thinking_info = ""
                        continue
                
                # Check column click
                if game.winner is None and game.turn == 0 and game.anim is None:
                    if 20 <= mx <= 20 + CELL_W * 7 and BOARD_TOP <= my <= BOARD_TOP + CELL_H * 6:
                        col = (mx - 20) // CELL_W
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
                    thinking_info = f"up to {max_think_time:.1f}s ({stop_mode})"
            else:
                game.anim = (col, target_row, color, frame)

        # Draw frame during thinking
        draw_board(screen, game, ai_thinking=ai_thinking, thinking_text=thinking_info)

        # AI turn.
        if ai_thinking and game.anim is None:
            col, sims_done, elapsed = agent.select_action(
                game.yellow, game.red, max_think_time=max_think_time, temperature=0.0
            )
            if sims_done == 0:
                thinking_info = f"model eval in {format_duration(elapsed)}"
            else:
                thinking_info = f"{sims_done:,} sims in {format_duration(elapsed)}"
            if find_row(game.yellow, col) is not None:
                game.turn = 1
                row = game.drop(col)
                if row is not None:
                    game.anim = (col, row, COLORS["yellow"], 0)
            ai_thinking = False

        clock.tick(FPS)

        draw_board(screen, game)
        clock.tick(FPS)

    pygame.quit()


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import CONFIG
    _DEFAULT_GUI_MODEL = "auto"
    _DEFAULT_GUI_DEVICE = CONFIG.device.python_device
except Exception:
    _DEFAULT_GUI_MODEL = "auto"
    _DEFAULT_GUI_DEVICE = "auto"


def main() -> None:
    p = argparse.ArgumentParser(description="Play Connect 4 vs Connect4Net")
    p.add_argument(
        "--model",
        default=_DEFAULT_GUI_MODEL,
        help="path to a .pt/.onnx model, or auto to use best_model.json",
    )
    p.add_argument("--device", default=_DEFAULT_GUI_DEVICE)
    p.add_argument(
        "--backend",
        choices=["auto", "onnx", "tensorrt", "torch"],
        default="auto",
        help="inference backend (auto: TensorRT/ONNX CUDA/PyTorch fallback)",
    )
    args = p.parse_args()
    try:
        run(args.model, args.device, args.backend)
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
