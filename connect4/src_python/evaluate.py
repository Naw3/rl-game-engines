import argparse
import sys
import os
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
            policy_logits, v, _ = model(state)
        
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
    args = parser.parse_args()

    print(f"[evaluate] Loading models on {args.device}...")
    m1 = Connect4Net()
    m1.load_state_dict(torch.load(args.model1, map_location="cpu", weights_only=True))
    m1.to(args.device)
    m1.eval()
    
    m2 = Connect4Net()
    m2.load_state_dict(torch.load(args.model2, map_location="cpu", weights_only=True))
    m2.to(args.device)
    m2.eval()
    
    # Play games (p1 plays first half of games, p2 plays second half)
    m1_wins = 0
    m2_wins = 0
    draws = 0
    
    half_games = args.games // 2
    
    print(f"[evaluate] Playing {args.games} games between M1 ({os.path.basename(args.model1)}) and M2 ({os.path.basename(args.model2)}) at Temp=0...")
    
    import time
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
