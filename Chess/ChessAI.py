"""
Handling the AI moves.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from ChessEngine import Move
import chess

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 4096)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return self.softmax(self.fc1(x))
    
model_path = os.path.join(os.path.dirname(__file__), "trained_models", "chess_ai_model.pth")
model = ChessNet()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
else:
    print("Không tìm thấy model, AI sẽ đánh ngẫu nhiên.")
    model = None

    
def board_to_tensor(board):
    piece_to_index = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
                      "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11}
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            tensor[piece_to_index[piece.symbol()]][row][col] = 1
    return torch.tensor(tensor).unsqueeze(0) 

def fen_to_vector(fen):
    vector = np.zeros((8, 8, 12), dtype=np.float32)
    board = chess.Board(fen)
    piece_symbols = "pnbrqkPNBRQK"

    for square, piece in board.piece_map().items():
        piece_index = piece_symbols.index(piece.symbol())
        row, col = divmod(square, 8)
        vector[row][col][piece_index] = 1

    return vector

def get_uci_notation(move):
    return f"{chr(move.start_col + 97)}{8 - move.start_row}{chr(move.end_col + 97)}{8 - move.end_row}"

def findBestMoveFromModel(game_state, valid_moves):
    try:
        fen_str = game_state.fen()
        input_tensor = torch.tensor(fen_to_vector(fen_str), dtype=torch.float32)
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 12, 8, 8]

        with torch.no_grad():
            output = model(input_tensor)
            move_index = torch.argmax(output, dim=1).item()  

        top_indices = torch.argsort(output, descending=True)[0] 

        for move_index in top_indices:
            start_square = move_index.item() // 64
            end_square = move_index.item() % 64
            move_uci = chess.Move(start_square, end_square).uci()

            for move in valid_moves:
                if move.getChessNotationUCI() == move_uci:
                    return move

        print(f"AI chọn nước không hợp lệ: {move_uci} ở vị trí FEN: {fen_str}")
        return random.choice(valid_moves)

    except Exception as e:
        print(f"Lỗi khi xử lý nước đi: {e}")
        return random.choice(valid_moves)



# piece_score = {"K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1}

# knight_scores = np.array([
#     [-5, -4, -3, -3, -3, -3, -4, -5],
#     [-4, -2, 0, 0, 0, 0, -2, -4],
#     [-3, 0, 1, 1.5, 1.5, 1, 0, -3],
#     [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
#     [-3, 0, 1.5, 2, 2, 1.5, 0, -3],
#     [-3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3],
#     [-4, -2, 0, 0.5, 0.5, 0, -2, -4],
#     [-5, -4, -3, -3, -3, -3, -4, -5]
# ])


# bishop_scores = np.array([
#     [-2, -1, -1, -1, -1, -1, -1, -2],
#     [-1, 0, 0, 0, 0, 0, 0, -1],
#     [-1, 0, 0.5, 1, 1, 0.5, 0, -1],
#     [-1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1],
#     [-1, 0, 1, 1, 1, 1, 0, -1],
#     [-1, 1, 1, 1, 1, 1, 1, -1],
#     [-1, 0.5, 0, 0, 0, 0, 0.5, -1],
#     [-2, -1, -1, -1, -1, -1, -1, -2]
# ])


# rook_scores = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0.5, 1, 1, 1, 1, 1, 1, 0.5],
#     [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
#     [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
#     [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
#     [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
#     [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
#     [0, 0, 0, 0.5, 0.5, 0, 0, 0]
# ])


# queen_scores = np.array([
#     [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
#     [-1, 0, 0, 0, 0, 0, 0, -1],
#     [-1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1],
#     [-0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
#     [0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
#     [-1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1],
#     [-1, 0, 0.5, 0, 0, 0, 0, -1],
#     [-2, -1, -1, -0.5, -0.5, -1, -1, -2]
# ])


# pawn_scores = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [5, 5, 5, 5, 5, 5, 5, 5],
#     [1, 1, 2, 3, 3, 2, 1, 1],
#     [0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5],
#     [0, 0, 0, 2, 2, 0, 0, 0],
#     [0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5],
#     [0.5, 1, 1, -2, -2, 1, 1, 0.5],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ])

# king_scores = np.array([
#     [-3, -4, -4, -5, -5, -4, -4, -3],
#     [-3, -4, -4, -5, -5, -4, -4, -3],
#     [-3, -4, -4, -5, -5, -4, -4, -3],
#     [-3, -4, -4, -5, -5, -4, -4, -3],
#     [-2, -3, -3, -4, -4, -3, -3, -2],
#     [-1, -2, -2, -2, -2, -2, -2, -1],
#     [2, 2, 0, 0, 0, 0, 2, 2],
#     [2, 3, 1, 0, 0, 1, 3, 2]
# ])


# piece_position_scores = {
#     "wN": knight_scores,
#     "bN": knight_scores[::-1],
#     "wB": bishop_scores,
#     "bB": bishop_scores[::-1],
#     "wQ": queen_scores,
#     "bQ": queen_scores[::-1],
#     "wR": rook_scores,
#     "bR": rook_scores[::-1],
#     "wp": pawn_scores,
#     "bp": pawn_scores[::-1],
#     "wK": np.zeros((8, 8)),  
#     "bK": np.zeros((8, 8))   
# }

# CHECKMATE = 10000
# STALEMATE = 0
# DEPTH = 4

# def get_depth_for_level(level):
#     """Trả về độ sâu tìm kiếm dựa trên cấp độ AI"""
#     if level == "easy":
#         return 2  
#     elif level == "hard":
#         return 4 
    
# def order_moves(game_state, valid_moves):
#     """
#     Sắp xếp các nước đi dựa trên giá trị của quân bị bắt (Capture Move ưu tiên).
#     """
#     def move_score(move):
#         target_square = game_state.board[move.end_row][move.end_col]
#         if target_square != "--":
#             return piece_score.get(target_square[1], 0)  
#         return 0

#     return sorted(valid_moves, key=move_score, reverse=True)


# def findBestMove(game_state, valid_moves, ai_level, return_queue):
#     """
#     Tìm kiếm tăng dần từ độ sâu 1 đến độ sâu tối đa để tìm nước đi tốt nhất.
#     """
#     global next_move
#     next_move = None
#     transposition_table.clear()  

#     max_depth = get_depth_for_level(ai_level)  

#     for depth in range(1, max_depth + 1):
#         findMoveNegaMaxAlphaBeta(game_state, valid_moves, depth, -CHECKMATE, CHECKMATE,
#                                  1 if game_state.white_to_move else -1)
    

#     return_queue.put(next_move)


# def quiescence_search(game_state, alpha, beta, turn_multiplier):
#     """
#     Tìm kiếm yên tĩnh để tránh cắt tỉa các nước đi ăn quân hoặc gây thay đổi lớn.
#     """
#     stand_pat = turn_multiplier * scoreBoard(game_state)

#     if stand_pat >= beta:
#         return beta
#     if alpha < stand_pat:
#         alpha = stand_pat

#     capture_moves = [move for move in game_state.getValidMoves() if game_state.board[move.end_row][move.end_col] != "--"]
#     capture_moves = order_moves(game_state, capture_moves)

#     for move in capture_moves:
#         game_state.makeMove(move)
#         score = -quiescence_search(game_state, -beta, -alpha, -turn_multiplier)
#         game_state.undoMove()

#         if score >= beta:
#             return beta
#         alpha = max(alpha, score)

#     return alpha


# transposition_table = {}

# def findMoveNegaMaxAlphaBeta(game_state, valid_moves, depth, alpha, beta, turn_multiplier):
#     """
#     NegaMax với Alpha-Beta Pruning, hỗ trợ Tìm kiếm Tăng Dần và Bảng băm.
#     """
#     global next_move

#     board_key = str(game_state.board) + str(game_state.white_to_move)
#     if board_key in transposition_table and transposition_table[board_key]['depth'] >= depth:
#         return transposition_table[board_key]['score']

#     if depth == 0 or len(valid_moves) == 0:
#         return turn_multiplier * scoreBoard(game_state)

#     valid_moves = order_moves(game_state, valid_moves)
#     max_score = -CHECKMATE  

#     for move in valid_moves:
#         game_state.makeMove(move)
#         next_moves = game_state.getValidMoves()

#         score = -findMoveNegaMaxAlphaBeta(game_state, next_moves, depth - 1, -beta, -alpha, -turn_multiplier)
#         game_state.undoMove()

#         if score > max_score:
#             max_score = score
#             if depth == DEPTH:  
#                 next_move = move

#         alpha = max(alpha, max_score)
#         if alpha >= beta:
#             break

#     transposition_table[board_key] = {"score": max_score, "depth": depth}
#     return max_score




# def scoreBoard(game_state):
#     """
#     Score the board. A positive score is good for white, a negative score is good for black.
#     """
#     score = 0
#     for row in range(8):
#         for col in range(8):
#             piece = game_state.board[row][col]
#             if piece != "--":
#                 piece_value = piece_score.get(piece[1], 0)
#                 position_score = piece_position_scores.get(piece, np.zeros((8, 8)))[row][col]
#                 if piece[0] == "w":
#                     score += piece_value + position_score
#                 else:
#                     score -= piece_value + position_score
#     return score


# def findRandomMove(valid_moves):
#     """
#     Picks and returns a random valid move.
#     """
#     return random.choice(valid_moves)

