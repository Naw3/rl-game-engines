import argparse
import json
import re
import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# Add src_python to path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import Connect4Net

# Connect 4 logic adapted for simple fast evaluation
ROWS = 6
COLS = 7
_EPOCH_MODEL_RE = re.compile(
    r"^(?P<base>.+?)_epoch(?P<epoch>\d+)(?P<ema>_ema)?$",
    re.IGNORECASE,
)


def latest_epoch_variant(model_path: Path) -> Path:
    """Map a stable raw/EMA path to its newest epoch-named checkpoint."""
    match = _EPOCH_MODEL_RE.match(model_path.stem)
    if match:
        base_stem = match.group("base")
        is_ema = bool(match.group("ema"))
    else:
        base_stem = model_path.stem.removesuffix("_ema")
        is_ema = model_path.stem.endswith("_ema")

    candidates: list[tuple[int, Path]] = []
    for candidate in model_path.parent.glob(f"{base_stem}_epoch*.pt"):
        candidate_match = _EPOCH_MODEL_RE.match(candidate.stem)
        if (
            candidate_match
            and candidate_match.group("base") == base_stem
            and bool(candidate_match.group("ema")) == is_ema
        ):
            candidates.append((int(candidate_match.group("epoch")), candidate))

    return max(candidates, key=lambda item: item[0])[1] if candidates else model_path

def get_valid_moves(board):
    return (board[0, :] == 0).nonzero()[0]

def play_move(board, col, player):
    # Find lowest empty row in col
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            return r
    return -1

def check_win(board, player):
    # Check horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            if board[r, c] == player and board[r, c+1] == player and board[r, c+2] == player and board[r, c+3] == player:
                return True
    # Check vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if board[r, c] == player and board[r+1, c] == player and board[r+2, c] == player and board[r+3, c] == player:
                return True
    # Check diagonal \
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if board[r, c] == player and board[r+1, c+1] == player and board[r+2, c+2] == player and board[r+3, c+3] == player:
                return True
    # Check diagonal /
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if board[r, c] == player and board[r-1, c+1] == player and board[r-2, c+2] == player and board[r-3, c+3] == player:
                return True
    return False


def check_win_bitboard(bitboard: int) -> bool:
    """Check a Connect 4 bitboard using the same layout as inference.py."""
    for shift in (1, 7, 6, 8):
        aligned = bitboard & (bitboard >> shift)
        if aligned & (aligned >> (2 * shift)):
            return True
    return False


def play_game_mcts(agent_p1, agent_p2, think_time: float, sims: int) -> int:
    """Play one game with the exact MCTS agent used by the GUI."""
    boards = {1: 0, -1: 0}
    agents = {1: agent_p1, -1: agent_p2}
    player = 1

    agent_p1.reset_cache()
    agent_p2.reset_cache()

    for _ in range(42):
        own = boards[player]
        opp = boards[-player]
        action, _, _ = agents[player].select_action(
            own,
            opp,
            sims=sims,
            max_think_time=think_time,
            temperature=0.0,
        )

        for row in range(6):
            bit = 1 << (action * 7 + row)
            if not ((own | opp) & bit):
                boards[player] |= bit
                break
        else:
            return -player

        if check_win_bitboard(boards[player]):
            return player
        player = -player

    return 0


def write_best_model_selection(
    selection_file: str | None,
    model1: str,
    model2: str,
    m1_wins: int,
    m2_wins: int,
    draws: int,
) -> Path:
    """Persist the MCTS winner for GUI auto-selection."""
    model1_path = Path(model1).resolve()
    model2_path = Path(model2).resolve()
    if m1_wins > m2_wins:
        selected = model1_path
    else:
        # A tie deliberately keeps the current/raw model (model2 in the
        # pipeline), because a tie is not evidence that EMA is better.
        selected = model2_path
    selected = latest_epoch_variant(selected)

    output = Path(selection_file) if selection_file else model2_path.parent / "best_model.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_onnx = selected.with_suffix(".onnx")
    payload = {
        "model_pt": os.path.relpath(selected, output.parent),
        "model_onnx": os.path.relpath(selected_onnx, output.parent) if selected_onnx.exists() else None,
        "selection": "model1" if m1_wins > m2_wins else "model2",
        "selection_method": "mcts_head_to_head",
        "model1": str(model1_path),
        "model2": str(model2_path),
        "model1_wins": m1_wins,
        "model2_wins": m2_wins,
        "draws": draws,
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"[evaluate] Best model selected by MCTS: {selected.name}")
    print(f"[evaluate] Selection manifest written to {output}")
    return output

def format_state(board, player_to_move):
    # Shape: (1, 3, 6, 7)
    state = np.zeros((1, 3, ROWS, COLS), dtype=np.float32)
    state[0, 0] = (board == player_to_move)
    state[0, 1] = (board == -player_to_move)
    state[0, 2] = 1.0 # constant turn plane
    return torch.from_numpy(state)

def play_game(model_p1, model_p2, device="cuda"):
    board = np.zeros((ROWS, COLS), dtype=int)
    players = [1, -1]
    models = {1: model_p1, -1: model_p2}
    
    turn = 0
    while True:
        player = players[turn % 2]
        model = models[player]
        
        valid_moves = get_valid_moves(board)
        if len(valid_moves) == 0:
            return 0 # Draw
            
        state = format_state(board, player).to(device)
        with torch.no_grad():
            policy_logits, v, _moves_left, _confidence = model(state)
        
        # Temp 0: argmax of policy over valid moves
        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        
        # Mask invalid
        mask = np.zeros(COLS, dtype=bool)
        mask[valid_moves] = True
        
        valid_policy = policy[mask]
        best_valid_idx = np.argmax(valid_policy)
        best_move = valid_moves[best_valid_idx]
        
        play_move(board, best_move, player)
        
        if check_win(board, player):
            return player
            
        turn += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True, help="Path to first model (.pt)")
    parser.add_argument("--model2", type=str, required=True, help="Path to second model (.pt)")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mcts", action="store_true",
        help="evaluate with the same Connect4Agent MCTS used by inference.py/gui.py",
    )
    parser.add_argument("--think-time", type=float, default=0.25, help="MCTS time budget per move in --mcts mode")
    parser.add_argument("--sims", type=int, default=0, help="MCTS simulation cap (0 = time budget only)")
    parser.add_argument(
        "--backend",
        choices=[
            "auto",
            "pytorch-cuda",
            "pytorch-cpu",
            "onnx-cuda",
            "tensorrt",
            "torch",
            "onnx",
        ],
        default="auto",
        help="inference backend for --mcts mode (auto uses config.py infer_backend)",
    )
    parser.add_argument("--selection-file", type=str, default=None, help="write the MCTS winner to this JSON file")
    args = parser.parse_args()

    if args.mcts:
        from inference import Connect4Agent

        print(
            f"[evaluate] MCTS head-to-head: {args.games} games, "
            f"{args.think_time:.3f}s/move, backend={args.backend}"
        )
        agent1 = Connect4Agent(args.model1, device=args.device, backend=args.backend, verbose=False)
        agent2 = Connect4Agent(args.model2, device=args.device, backend=args.backend, verbose=False)
        m1_wins = 0
        m2_wins = 0
        draws = 0
        half_games = args.games // 2
        t0 = time.time()

        for _ in range(half_games):
            result = play_game_mcts(agent1, agent2, args.think_time, args.sims)
            if result == 1:
                m1_wins += 1
            elif result == -1:
                m2_wins += 1
            else:
                draws += 1

        for _ in range(args.games - half_games):
            result = play_game_mcts(agent2, agent1, args.think_time, args.sims)
            if result == 1:
                m2_wins += 1
            elif result == -1:
                m1_wins += 1
            else:
                draws += 1

        elapsed = time.time() - t0
        print(f"[evaluate] MCTS results after {args.games} games ({elapsed:.1f}s):")
        print(f"  M1 ({os.path.basename(args.model1)}) Wins: {m1_wins}")
        print(f"  M2 ({os.path.basename(args.model2)}) Wins: {m2_wins}")
        print(f"  Draws: {draws}")
        write_best_model_selection(
            args.selection_file, args.model1, args.model2, m1_wins, m2_wins, draws
        )
        return

    print(f"[evaluate] Loading models on {args.device}...")
    m1 = Connect4Net.load(args.model1)
    m1.to(args.device)
    m1.eval()
    
    m2 = Connect4Net.load(args.model2)
    m2.to(args.device)
    m2.eval()
    
    # Play games (p1 plays first half of games, p2 plays second half)
    m1_wins = 0
    m2_wins = 0
    draws = 0
    
    half_games = args.games // 2
    
    print(f"[evaluate] Playing {args.games} games between M1 ({os.path.basename(args.model1)}) and M2 ({os.path.basename(args.model2)}) at Temp=0...")
    
    t0 = time.time()
    
    # M1 is player 1
    for i in range(half_games):
        res = play_game(m1, m2, args.device)
        if res == 1:
            m1_wins += 1
        elif res == -1:
            m2_wins += 1
        else:
            draws += 1
            
    # M2 is player 1
    for i in range(args.games - half_games):
        res = play_game(m2, m1, args.device)
        if res == 1:
            m2_wins += 1
        elif res == -1:
            m1_wins += 1
        else:
            draws += 1
            
    elapsed = time.time() - t0
    
    print(f"[evaluate] Results after {args.games} games ({elapsed:.1f}s):")
    print(f"  M1 ({os.path.basename(args.model1)}) Wins: {m1_wins} ({(m1_wins/args.games)*100:.1f}%)")
    print(f"  M2 ({os.path.basename(args.model2)}) Wins: {m2_wins} ({(m2_wins/args.games)*100:.1f}%)")
    print(f"  Draws: {draws} ({(draws/args.games)*100:.1f}%)")
    
    if m1_wins > m2_wins:
        print("  => M1 is the winner!")
    elif m2_wins > m1_wins:
        print("  => M2 is the winner!")
    else:
        print("  => It's a tie!")

if __name__ == "__main__":
    main()
