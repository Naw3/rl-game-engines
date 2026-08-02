"""
gui.py — Modern Connect 4 application with Pygame UI, Main Menu, and async MCTS Agent.

Features:
- Instantaneous non-blocking Pygame startup with smooth background agent loading.
- Main Menu for selecting game modes: Human vs AI, AI vs AI, Human vs Human.
- Live Win Probability evaluation bar and real-time MCTS metrics.
- Dark glassmorphism aesthetic with animated piece drops and interactive UI.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CONFIG
from src_python.inference import Connect4Agent, _check_win_bitboard, get_legal_actions_mask

# --- Palette & Design Tokens ---
COLOR_BG = (18, 20, 32)
COLOR_CARD = (30, 34, 52)
COLOR_CARD_BORDER = (50, 58, 88)
COLOR_BOARD_BG = (28, 42, 90)
COLOR_BOARD_INNER = (40, 60, 140)
COLOR_HOLE = (18, 20, 32)

COLOR_RED = (235, 65, 65)
COLOR_RED_GLOW = (255, 100, 100)
COLOR_YELLOW = (245, 205, 45)
COLOR_YELLOW_GLOW = (255, 225, 90)

COLOR_TEXT_MAIN = (240, 244, 255)
COLOR_TEXT_MUTED = (140, 150, 180)
COLOR_ACCENT = (90, 140, 255)
COLOR_ACCENT_HOVER = (120, 165, 255)
COLOR_SUCCESS = (65, 210, 120)

WINDOW_W = 900
WINDOW_H = 750
FPS = 60

COLS, ROWS = 7, 6


def find_winning_line(bitboard: int) -> list[tuple[int, int]]:
    """Return list of 4 (col, row) coordinates forming the winning line."""
    # Horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            bits = [(c + i, r) for i in range(4)]
            if all((bitboard & (1 << (bc * 7 + br))) for bc, br in bits):
                return bits
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            bits = [(c, r + i) for i in range(4)]
            if all((bitboard & (1 << (bc * 7 + br))) for bc, br in bits):
                return bits
    # Diagonal \
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            bits = [(c + i, r + i) for i in range(4)]
            if all((bitboard & (1 << (bc * 7 + br))) for bc, br in bits):
                return bits
    # Diagonal /
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            bits = [(c + i, r - i) for i in range(4)]
            if all((bitboard & (1 << (bc * 7 + br))) for bc, br in bits):
                return bits
    return []


class GameState:
    """Connect 4 Game Logic (Bitboard representation)."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.own = 0  # Red / Player 1 (drops first)
        self.opp = 0  # Yellow / Player 2
        self.turn = 1  # 1 for Red, -1 for Yellow
        self.winner: int | None = None  # 1, -1, or 0 (draw)
        self.winning_line: list[tuple[int, int]] = []
        self.history: list[int] = []
        self.move_count = 0

    def is_legal(self, col: int) -> bool:
        if self.winner is not None or not (0 <= col < COLS):
            return False
        occ = self.own | self.opp
        return (occ & (1 << (col * 7 + 5))) == 0

    def drop(self, col: int) -> int | None:
        if not self.is_legal(col):
            return None

        occ = self.own | self.opp
        col_shift = col * 7
        row_bit = None
        row_idx = None
        for r in range(ROWS):
            bit = 1 << (col_shift + r)
            if (occ & bit) == 0:
                row_bit = bit
                row_idx = r
                break

        if row_bit is None or row_idx is None:
            return None

        if self.turn == 1:
            self.own |= row_bit
            if _check_win_bitboard(self.own):
                self.winner = 1
                self.winning_line = find_winning_line(self.own)
        else:
            self.opp |= row_bit
            if _check_win_bitboard(self.opp):
                self.winner = -1
                self.winning_line = find_winning_line(self.opp)

        self.history.append(col)
        self.move_count += 1

        if self.winner is None and (self.own | self.opp).bit_count() == 42:
            self.winner = 0

        if self.winner is None:
            self.turn = -self.turn

        return row_idx


class Connect4App:
    def __init__(self, model_path: str | None, device_str: str, backend: str) -> None:
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Connect 4 — Neural MCTS Engine")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 36, bold=True)
        self.font_header = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 16)
        self.font_small = pygame.font.SysFont("Segoe UI", 13)

        # Agent & Async Loading
        self.model_path = model_path
        self.device_str = device_str
        self.backend = backend
        self.agent: Connect4Agent | None = None
        self.agent_loading = True
        self.agent_error: str | None = None

        # Start async agent loading thread
        threading.Thread(target=self._load_agent_async, daemon=True).start()

        # Navigation & Mode State
        self.mode = "MENU"  # "MENU", "GAME"
        self.game_type = "HUMAN_VS_AI"  # "HUMAN_VS_AI", "AI_VS_AI", "HUMAN_VS_HUMAN"
        self.think_time = float(getattr(CONFIG.infer, "max_think_time", 1.0))
        
        self.game = GameState()
        
        # Piece animation state: list of (col, target_row, color, current_y)
        self.animating_pieces: list[dict] = []
        
        # MCTS background thinking thread
        self.ai_thinking = False
        self.last_mcts_stats = {
            "time": "0ms",
            "sims": "0",
            "reused": "0",
            "sps": "0",
            "confidence": "0%",
            "value": 0.0,
            "cache_hits": "0%",
        }
        self.win_probability = 0.5  # 0.5 = neutral, 1.0 = Red win, 0.0 = Yellow win

        # UI Dimensions
        self.board_rect = pygame.Rect(260, 110, 600, 540)
        self.cell_w = self.board_rect.width // COLS
        self.cell_h = self.board_rect.height // ROWS

    def _load_agent_async(self) -> None:
        try:
            agent = Connect4Agent(
                self.model_path,
                device=self.device_str,
                backend=self.backend,
                verbose=True,
            )
            self.agent = agent
        except Exception as e:
            self.agent_error = str(e)
        finally:
            self.agent_loading = False

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self._handle_event(event)

            # Update & Render
            self._update(dt)
            self._render()
            pygame.display.flip()

        pygame.quit()

    def _handle_event(self, event: pygame.event.Event) -> None:
        if self.mode == "MENU":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Game mode buttons
                if pygame.Rect(270, 240, 360, 55).collidepoint(mx, my):
                    self.game_type = "HUMAN_VS_AI"
                    self._start_game()
                elif pygame.Rect(270, 315, 360, 55).collidepoint(mx, my):
                    self.game_type = "AI_VS_AI"
                    self._start_game()
                elif pygame.Rect(270, 390, 360, 55).collidepoint(mx, my):
                    self.game_type = "HUMAN_VS_HUMAN"
                    self._start_game()
                # Think time adjustment
                elif pygame.Rect(270, 485, 110, 40).collidepoint(mx, my):
                    self.think_time = 0.25
                elif pygame.Rect(395, 485, 110, 40).collidepoint(mx, my):
                    self.think_time = 1.0
                elif pygame.Rect(520, 485, 110, 40).collidepoint(mx, my):
                    self.think_time = 3.0

        elif self.mode == "GAME":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.game.reset()
                    if self.agent:
                        self.agent.reset_cache()
                    self.win_probability = 0.5
                elif event.key == pygame.K_m:
                    self.mode = "MENU"
                    self.game.reset()
                    if self.agent:
                        self.agent.reset_cache()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                
                # Check Header buttons (Menu / Reset)
                if pygame.Rect(20, 20, 110, 36).collidepoint(mx, my):
                    self.mode = "MENU"
                    self.game.reset()
                    if self.agent:
                        self.agent.reset_cache()
                elif pygame.Rect(140, 20, 100, 36).collidepoint(mx, my):
                    self.game.reset()
                    if self.agent:
                        self.agent.reset_cache()
                    self.win_probability = 0.5
                    
                # Check Column clicks for Human turn
                elif self.game.winner is None and not self.ai_thinking and self._is_human_turn():
                    if self.board_rect.collidepoint(mx, my):
                        col = (mx - self.board_rect.x) // self.cell_w
                        self._trigger_move(col)

    def _start_game(self) -> None:
        self.game.reset()
        if self.agent:
            self.agent.reset_cache()
        self.win_probability = 0.5
        self.mode = "GAME"

    def _is_human_turn(self) -> bool:
        if self.game_type == "HUMAN_VS_HUMAN":
            return True
        if self.game_type == "HUMAN_VS_AI":
            return self.game.turn == 1  # Human plays Red (P1)
        return False  # AI vs AI

    def _trigger_move(self, col: int) -> None:
        color = COLOR_RED if self.game.turn == 1 else COLOR_YELLOW
        target_row = self.game.drop(col)
        if target_row is not None:
            # Add drop animation
            start_y = self.board_rect.y - self.cell_h
            target_y = self.board_rect.y + (ROWS - 1 - target_row) * self.cell_h + self.cell_h // 2
            self.animating_pieces.append({
                "col": col,
                "target_row": target_row,
                "target_y": target_y,
                "current_y": float(start_y),
                "speed": 0.0,
                "color": color,
            })
            # Immediate win probability update on game over
            if self.game.winner is not None:
                if self.game.winner == 1:
                    self.win_probability = 0.98
                elif self.game.winner == -1:
                    self.win_probability = 0.02
                else:
                    self.win_probability = 0.50

    def _update(self, dt: float) -> None:
        # Update piece drop animations
        for p in list(self.animating_pieces):
            p["speed"] += 2800.0 * dt
            p["current_y"] += p["speed"] * dt
            if p["current_y"] >= p["target_y"]:
                p["current_y"] = p["target_y"]
                self.animating_pieces.remove(p)

        # AI Turn Trigger
        if (
            self.mode == "GAME"
            and self.game.winner is None
            and not self.ai_thinking
            and not self.agent_loading
            and self.agent is not None
            and not self._is_human_turn()
            and not self.animating_pieces
        ):
            self.ai_thinking = True
            threading.Thread(target=self._run_ai_think, daemon=True).start()

    def _run_ai_think(self) -> None:
        if self.agent is None:
            self.ai_thinking = False
            return

        ai_player = self.game.turn  # 1 for Red, 2 (-1) for Yellow
        own = self.game.own if ai_player == 1 else self.game.opp
        opp = self.game.opp if ai_player == 1 else self.game.own
        
        hits_before = self.agent.cache_hits
        misses_before = self.agent.cache_misses

        t0 = time.perf_counter()
        action, sims_done, elapsed = self.agent.select_action(
            own,
            opp,
            max_think_time=self.think_time,
            temperature=0.0,
        )

        hits_delta = self.agent.cache_hits - hits_before
        misses_delta = self.agent.cache_misses - misses_before
        tot_evals = max(1, hits_delta + misses_delta)
        hit_pct = (hits_delta / tot_evals) * 100.0

        # Extract search Q-value from tree_root if available
        root = getattr(self.agent, "tree_root", None)
        if root is not None and hasattr(root, "N") and root.N.sum() > 0:
            best_a = int(np.argmax(root.N))
            val = float(root.W[best_a] / max(1, root.N[best_a]))
        else:
            _p, val, _c = self.agent.evaluate_position(own, opp)

        # val is from perspective of ai_player
        if ai_player == 1:
            win_prob = (val + 1.0) / 2.0
        else:
            win_prob = ((-val) + 1.0) / 2.0

        _p, _val_nn, _c = self.agent.evaluate_position(own, opp)
        reused_sims = getattr(self.agent, "last_reused_sims", 0)

        self.last_mcts_stats = {
            "time": f"{max(1, int(elapsed * 1000))}ms",
            "sims": f"{sims_done:,}",
            "reused": f"{int(reused_sims):,}",
            "sps": f"{int(sims_done / max(0.001, elapsed)):,}",
            "confidence": f"{_c * 100:.1f}%" if _c is not None else "N/A",
            "value": float(val),
            "cache_hits": f"{hit_pct:.1f}%",
        }

        # Perform AI move
        self._trigger_move(action)
        if self.game.winner is None:
            self.win_probability = float(np.clip(win_prob, 0.02, 0.98))
        self.ai_thinking = False

    def _render(self) -> None:
        self.screen.fill(COLOR_BG)

        if self.mode == "MENU":
            self._render_menu()
        elif self.mode == "GAME":
            self._render_game()

    def _render_menu(self) -> None:
        # Title Card
        title_surf = self.font_title.render("CONNECT 4 NEURAL ENGINE", True, COLOR_TEXT_MAIN)
        sub_surf = self.font_body.render("AlphaZero MCTS AI with ONNX / CUDA Hardware Acceleration", True, COLOR_TEXT_MUTED)
        self.screen.blit(title_surf, title_surf.get_rect(center=(WINDOW_W // 2, 100)))
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(WINDOW_W // 2, 145)))

        # Agent Loading Status Card
        status_rect = pygame.Rect(270, 175, 360, 45)
        pygame.draw.rect(self.screen, COLOR_CARD, status_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, status_rect, 1, border_radius=10)

        if self.agent_loading:
            dots = "." * (int(time.time() * 4) % 4)
            st_text = f"Initializing Neural Engine{dots}"
            st_color = COLOR_YELLOW
        elif self.agent_error:
            st_text = f"Engine Error: {self.agent_error[:25]}..."
            st_color = COLOR_RED
        else:
            st_text = "Engine Ready (CUDA / ONNX active)"
            st_color = COLOR_SUCCESS

        st_surf = self.font_body.render(st_text, True, st_color)
        self.screen.blit(st_surf, st_surf.get_rect(center=status_rect.center))

        # Game Mode Buttons
        modes = [
            ("HUMAN_VS_AI", "[VS AI]  Human vs AI", "Play Red pieces against the Neural MCTS AI"),
            ("AI_VS_AI", "[AI vs AI]  AI vs AI", "Watch two AI agents battle with MCTS search"),
            ("HUMAN_VS_HUMAN", "[VS P2]  Human vs Human", "Pass and play with a local friend"),
        ]

        for idx, (m_code, m_label, m_desc) in enumerate(modes):
            y_pos = 240 + idx * 75
            btn_rect = pygame.Rect(270, y_pos, 360, 60)
            is_hover = btn_rect.collidepoint(pygame.mouse.get_pos())
            
            bg_col = COLOR_ACCENT if is_hover else COLOR_CARD
            border_col = COLOR_ACCENT_HOVER if is_hover else COLOR_CARD_BORDER

            pygame.draw.rect(self.screen, bg_col, btn_rect, border_radius=12)
            pygame.draw.rect(self.screen, border_col, btn_rect, 2, border_radius=12)

            lbl_surf = self.font_header.render(m_label, True, COLOR_TEXT_MAIN)
            self.screen.blit(lbl_surf, (btn_rect.x + 20, btn_rect.y + 10))
            
            desc_surf = self.font_small.render(m_desc, True, COLOR_TEXT_MUTED if not is_hover else COLOR_TEXT_MAIN)
            self.screen.blit(desc_surf, (btn_rect.x + 20, btn_rect.y + 35))

        # Think Time Options
        tt_lbl = self.font_header.render("AI Search Time per Move:", True, COLOR_TEXT_MAIN)
        self.screen.blit(tt_lbl, (270, 460))

        options = [(0.25, "0.25s (Fast)"), (1.0, "1.0s (Normal)"), (3.0, "3.0s (Deep)")]
        for idx, (val, text) in enumerate(options):
            x_pos = 270 + idx * 125
            opt_rect = pygame.Rect(x_pos, 495, 110, 40)
            selected = abs(self.think_time - val) < 0.05
            
            bg_c = COLOR_ACCENT if selected else COLOR_CARD
            bd_c = COLOR_ACCENT_HOVER if selected else COLOR_CARD_BORDER

            pygame.draw.rect(self.screen, bg_c, opt_rect, border_radius=8)
            pygame.draw.rect(self.screen, bd_c, opt_rect, 2 if selected else 1, border_radius=8)

            t_surf = self.font_body.render(text, True, COLOR_TEXT_MAIN)
            self.screen.blit(t_surf, t_surf.get_rect(center=opt_rect.center))

    def _render_game(self) -> None:
        # Header Controls
        btn_menu = pygame.Rect(20, 20, 110, 36)
        btn_reset = pygame.Rect(140, 20, 100, 36)

        for btn, label in [(btn_menu, "← Menu"), (btn_reset, "↺ Reset")]:
            is_hover = btn.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(self.screen, COLOR_ACCENT if is_hover else COLOR_CARD, btn, border_radius=8)
            pygame.draw.rect(self.screen, COLOR_CARD_BORDER, btn, 1, border_radius=8)
            txt = self.font_body.render(label, True, COLOR_TEXT_MAIN)
            self.screen.blit(txt, txt.get_rect(center=btn.center))

        # Turn & Status Banner
        if self.game.winner is not None:
            if self.game.winner == 1:
                status_msg, status_col = "★ RED WINS VICTORY! ★", COLOR_RED
            elif self.game.winner == -1:
                status_msg, status_col = "★ YELLOW WINS VICTORY! ★", COLOR_YELLOW
            else:
                status_msg, status_col = "DRAW GAME!", COLOR_TEXT_MUTED
        elif self.ai_thinking:
            status_msg, status_col = "AI Thinking...", COLOR_YELLOW
        else:
            turn_name = "Red (Player 1)" if self.game.turn == 1 else "Yellow (Player 2)"
            status_msg, status_col = f"Turn: {turn_name}", COLOR_RED if self.game.turn == 1 else COLOR_YELLOW

        status_surf = self.font_header.render(status_msg, True, status_col)
        self.screen.blit(status_surf, (260, 25))

        # Left Panel (Win Bar + Stats)
        self._render_stats_panel()

        # Board Drawing
        self._render_board()

    def _render_stats_panel(self) -> None:
        panel_rect = pygame.Rect(20, 80, 220, 630)
        pygame.draw.rect(self.screen, COLOR_CARD, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, panel_rect, 1, border_radius=12)

        # Win Probability Meter
        lbl = self.font_header.render("Win Probability", True, COLOR_TEXT_MAIN)
        self.screen.blit(lbl, (35, 95))

        prob_rect = pygame.Rect(35, 130, 190, 24)
        pygame.draw.rect(self.screen, COLOR_YELLOW, prob_rect, border_radius=6)
        
        red_w = int(prob_rect.width * self.win_probability)
        if red_w > 0:
            red_rect = pygame.Rect(prob_rect.x, prob_rect.y, red_w, prob_rect.height)
            pygame.draw.rect(self.screen, COLOR_RED, red_rect, border_top_left_radius=6, border_bottom_left_radius=6)

        pct_red = int(self.win_probability * 100)
        pct_yel = 100 - pct_red
        txt_prob = self.font_small.render(f"Red {pct_red}%  |  Yellow {pct_yel}%", True, COLOR_TEXT_MAIN)
        self.screen.blit(txt_prob, txt_prob.get_rect(center=(prob_rect.centerx, prob_rect.centery)))

        # MCTS Statistics List
        y_off = 180
        stats = [
            ("Move Time", self.last_mcts_stats["time"]),
            ("New Sims", self.last_mcts_stats["sims"]),
            ("Subtree Reused", self.last_mcts_stats["reused"]),
            ("Search Speed", f"{self.last_mcts_stats['sps']} s/s"),
            ("Cache Hit Rate", self.last_mcts_stats["cache_hits"]),
            ("NN Confidence", self.last_mcts_stats["confidence"]),
        ]

        hdr_stats = self.font_header.render("MCTS Engine Stats", True, COLOR_TEXT_MAIN)
        self.screen.blit(hdr_stats, (35, y_off))
        y_off += 35

        for k, v in stats:
            k_s = self.font_small.render(k, True, COLOR_TEXT_MUTED)
            v_s = self.font_body.render(v, True, COLOR_TEXT_MAIN)
            self.screen.blit(k_s, (35, y_off))
            self.screen.blit(v_s, (35, y_off + 16))
            y_off += 45

    def _render_board(self) -> None:
        # Draw main board shadow & background
        pygame.draw.rect(self.screen, COLOR_BOARD_BG, self.board_rect, border_radius=16)
        pygame.draw.rect(self.screen, COLOR_BOARD_INNER, self.board_rect, 3, border_radius=16)

        # Column Hover Guide
        mx, my = pygame.mouse.get_pos()
        if self.board_rect.collidepoint(mx, my) and self.game.winner is None and not self.ai_thinking and self._is_human_turn():
            col = (mx - self.board_rect.x) // self.cell_w
            hover_rect = pygame.Rect(
                self.board_rect.x + col * self.cell_w,
                self.board_rect.y,
                self.cell_w,
                self.board_rect.height,
            )
            pygame.draw.rect(self.screen, (255, 255, 255, 15), hover_rect, border_radius=10)

        # Draw Grid Holes & Placed Pieces
        for c in range(COLS):
            for r in range(ROWS):
                center_x = self.board_rect.x + c * self.cell_w + self.cell_w // 2
                center_y = self.board_rect.y + (ROWS - 1 - r) * self.cell_h + self.cell_h // 2
                radius = min(self.cell_w, self.cell_h) // 2 - 6

                is_animating = any(p["col"] == c and p.get("target_row") == r for p in self.animating_pieces)

                bit = 1 << (c * 7 + r)
                if self.game.own & bit and not is_animating:
                    pygame.draw.circle(self.screen, COLOR_RED, (center_x, center_y), radius)
                    pygame.draw.circle(self.screen, COLOR_RED_GLOW, (center_x, center_y), radius, 2)
                elif self.game.opp & bit and not is_animating:
                    pygame.draw.circle(self.screen, COLOR_YELLOW, (center_x, center_y), radius)
                    pygame.draw.circle(self.screen, COLOR_YELLOW_GLOW, (center_x, center_y), radius, 2)
                else:
                    pygame.draw.circle(self.screen, COLOR_HOLE, (center_x, center_y), radius)
                    pygame.draw.circle(self.screen, (40, 48, 75), (center_x, center_y), radius, 1)

        # Render Drop Animations
        for p in self.animating_pieces:
            center_x = self.board_rect.x + p["col"] * self.cell_w + self.cell_w // 2
            center_y = int(p["current_y"])
            radius = min(self.cell_w, self.cell_h) // 2 - 6
            glow_col = COLOR_RED_GLOW if p["color"] == COLOR_RED else COLOR_YELLOW_GLOW
            pygame.draw.circle(self.screen, p["color"], (center_x, center_y), radius)
            pygame.draw.circle(self.screen, glow_col, (center_x, center_y), radius, 2)

        # Draw Winning Line Pulse Rings
        if self.game.winner is not None and self.game.winning_line:
            pulse = math.sin(time.time() * 10.0) * 0.5 + 0.5
            gold_val = int(180 + 75 * pulse)
            ring_col = (255, gold_val, 40)
            for wc, wr in self.game.winning_line:
                center_x = self.board_rect.x + wc * self.cell_w + self.cell_w // 2
                center_y = self.board_rect.y + (ROWS - 1 - wr) * self.cell_h + self.cell_h // 2
                radius = min(self.cell_w, self.cell_h) // 2 - 2
                pygame.draw.circle(self.screen, ring_col, (center_x, center_y), radius, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect 4 Neural Engine GUI")
    parser.add_argument("--model", type=str, default=None, help="Path to PyTorch or ONNX model")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cuda, cpu, auto)")
    parser.add_argument("--backend", type=str, default="auto", help="Inference backend (onnx-cuda, tensorrt, pytorch, auto)")
    args = parser.parse_args()

    app = Connect4App(args.model, args.device, args.backend)
    app.run()


if __name__ == "__main__":
    main()
