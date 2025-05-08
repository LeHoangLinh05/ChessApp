# Chess_AI1.py (Sửa đổi để sử dụng Model NNUE làm hàm đánh giá tĩnh)

from collections import OrderedDict
from operator import itemgetter
# import pandas as pd # Không cần pandas nữa trừ khi dùng cho việc khác
import numpy as np
import tensorflow as tf
import random
import json
import chess
import time
from ChessEngine import * # Đảm bảo bạn import đúng GameState và Move

# Biến toàn cục cho Transposition Table
transposition_table = {} # Khởi tạo TT
# (Tùy chọn) Các biến để theo dõi hiệu quả TT
tt_hits = 0
tt_probes = 0

# Định nghĩa các cờ cho TT entry
TT_EXACT = 0
TT_LOWERBOUND = 1 # Alpha, fail-high
TT_UPPERBOUND = 2 # Beta, fail-low

# --- Giữ lại các hằng số và bảng PST (Có thể không dùng PST nữa nếu chỉ dùng NNUE) ---
# WEIGHT_MATERIAL_PST = 1 # Có thể không cần thiết
CHECKMATE_SCORE = 30000 # Giữ lại điểm Checkmate/Stalemate
STALEMATE_SCORE = 0.0
MAX_PLY = 60 # Giới hạn độ sâu ply để tránh vòng lặp vô hạn
MAX_QS_PLY = 3
MAX_OPENING_MOVES_IN_AI = 8
# --- ĐƯỜNG DẪN TỚI MÔ HÌNH NNUE ĐÃ LƯU ---
path_to_nnue_model = 'D:/AI/Chess/ChessApp/Chess/model/train/my_chess_eval_model1.h5' # <-- THAY ĐỔI ĐƯỜNG DẪN .h5 CỦA BẠN
OPENING_BOOK_PATH = "D:/AI/Chess/ChessApp/Chess/pgnopening_book_from_pgn.json"
opening_book_loaded = {}
try:
    with open(OPENING_BOOK_PATH, 'r') as f:
        opening_book_loaded = json.load(f)
    print(f"Sách khai cuộc đã tải từ: {OPENING_BOOK_PATH}")
except FileNotFoundError:
    print(f"Không tìm thấy file sách khai cuộc: {OPENING_BOOK_PATH}. AI sẽ không dùng sách.")
except Exception as e:
    print(f"Lỗi khi tải sách khai cuộc: {e}")

# --- Tải Mô hình NNUE (Keras) ---
global model_nnue # Đổi tên biến model
try:
    # Sử dụng tf.keras.models.load_model cho file .h5
    model_nnue = tf.keras.models.load_model(path_to_nnue_model)
    print(f"Model NNUE đã được tải thành công từ: {path_to_nnue_model}")
    # Kiểm tra sơ bộ model (có thể thêm kiểm tra input/output shape nếu muốn)
    model_nnue.summary() # In cấu trúc model

except Exception as e:
    print(f"Lỗi nghiêm trọng khi tải model NNUE từ '{path_to_nnue_model}': {e}")
    raise RuntimeError(f"Không thể tải model NNUE từ '{path_to_nnue_model}'") from e

# --- Các bảng PST và hàm phụ trợ (Giữ lại phòng khi cần fallback hoặc kết hợp) ---
# ... (Giữ nguyên các mảng pawn_white_eval, knight_white_eval,...) ...
pawn_white_eval = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
                            [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
                            [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                            [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
                            [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], float)

pawn_black_eval = pawn_white_eval[::-1]

knight_white_eval = np.array([[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
                              [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
                              [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
                              [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
                              [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
                              [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
                              [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
                              [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]], float)

knight_black_eval = knight_white_eval[::-1]

bishop_white_eval = np.array([[-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
                              [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                              [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
                              [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
                              [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
                              [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
                              [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
                              [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]], float)

bishop_black_eval = bishop_white_eval[::-1]

rook_white_eval = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]], float)

rook_black_eval = rook_white_eval[::-1]

queen_white_eval = np.array([[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
                             [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                             [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                             [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                             [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                             [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                             [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
                             [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]], float)

queen_black_eval = queen_white_eval[::-1]

king_white_eval = np.array([[-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                            [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
                            [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
                            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
                            [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]], float)

king_black_eval = king_white_eval[::-1]

# Hàm square_to_coord giữ nguyên (nếu hàm get_piece_value_pst dùng PST)
def square_to_coord(square):
    # ... (code giữ nguyên) ...
    return {0: (7, 0), 1: (7, 1), 2: (7, 2), 3: (7, 3), 4: (7, 4), 5: (7, 5), 6: (7, 6), 7: (7, 7),
            8: (6, 0), 9: (6, 1), 10: (6, 2), 11: (6, 3), 12: (6, 4), 13: (6, 5), 14: (6, 6), 15: (6, 7),
            16: (5, 0), 17: (5, 1), 18: (5, 2), 19: (5, 3), 20: (5, 4), 21: (5, 5), 22: (5, 6), 23: (5, 7),
            24: (4, 0), 25: (4, 1), 26: (4, 2), 27: (4, 3), 28: (4, 4), 29: (4, 5), 30: (4, 6), 31: (4, 7),
            32: (3, 0), 33: (3, 1), 34: (3, 2), 35: (3, 3), 36: (3, 4), 37: (3, 5), 38: (3, 6), 39: (3, 7),
            40: (2, 0), 41: (2, 1), 42: (2, 2), 43: (2, 3), 44: (2, 4), 45: (2, 5), 46: (2, 6), 47: (2, 7),
            48: (1, 0), 49: (1, 1), 50: (1, 2), 51: (1, 3), 52: (1, 4), 53: (1, 5), 54: (1, 6), 55: (1, 7),
            56: (0, 0), 57: (0, 1), 58: (0, 2), 59: (0, 3), 60: (0, 4), 61: (0, 5), 62: (0, 6), 63: (0, 7)}[square]


# Hàm get_piece_value_pst (Đổi tên để phân biệt với hàm dùng NNUE)
# Hàm này chỉ tính điểm chất + PST (có thể dùng làm fallback)
def get_piece_value_pst(piece, square):
    # ... (Code hàm get_piece_value cũ giữ nguyên) ...
    if piece == '--': return 0.0
    try: x, y = square_to_coord(square)
    except KeyError: return 0.0
    piece_color = piece[0]; piece_type = piece[1]; piece_type_upper = piece_type.upper()
    sign = 1.0 if piece_color == 'w' else -1.0
    base_value = 0.0; pst_value = 0.0
    try:
        if piece_type_upper == 'P': base_value = 10.0; pst_value = pawn_white_eval[x][y] if piece_color == 'w' else pawn_black_eval[x][y]
        elif piece_type_upper == 'N': base_value = 30.0; pst_value = knight_white_eval[x][y] if piece_color == 'w' else knight_black_eval[x][y]
        elif piece_type_upper == 'B': base_value = 30.0; pst_value = bishop_white_eval[x][y] if piece_color == 'w' else bishop_black_eval[x][y]
        elif piece_type_upper == 'R': base_value = 50.0; pst_value = rook_white_eval[x][y] if piece_color == 'w' else rook_black_eval[x][y]
        elif piece_type_upper == 'Q': base_value = 90.0; pst_value = queen_white_eval[x][y] if piece_color == 'w' else queen_black_eval[x][y]
        elif piece_type_upper == 'K': base_value = 900.0; pst_value = king_white_eval[x][y] if piece_color == 'w' else king_black_eval[x][y]
        else: return 0.0
    except IndexError: return 0.0
    except Exception: return 0.0
    return sign * (base_value + pst_value)

piece_simple_value = {'P': 10, 'N': 30, 'B': 30, 'R': 50, 'Q': 90, 'K': 900,
                      'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 900}

def score_move_mvv_lva(move: Move, gs: GameState):
    score = 0
    if move.piece_captured != '--':  # Là nước bắt quân
        # Lấy ký hiệu quân từ board, không phải từ move.piece_moved/captured nếu chúng lưu trữ khác
        # Giả sử move.piece_moved là 'wP', move.piece_captured là 'bN'
        attacker_type = move.piece_moved[1].upper()  # 'P'
        victim_type = move.piece_captured[1].upper()  # 'N'

        # Ưu tiên bắt quân giá trị cao
        score += piece_simple_value.get(victim_type, 0) * 10
        # Trừ điểm nếu quân tấn công giá trị cao (ưu tiên quân giá trị thấp tấn công)
        score -= piece_simple_value.get(attacker_type, 0)
    return score

def is_check_move(gs_original: GameState, move: Move):
    gs_copy = gs_original.copy() # Làm việc trên bản sao để không ảnh hưởng gs gốc
    gs_copy.makeMove(move)
    # Giả sử gs_copy.king_in_check() trả về True nếu vua của đối phương bị chiếu
    # (tức là, vua của người chơi mà gs_copy.white_to_move đang là False)
    # Hoặc bạn có hàm gs_copy.is_king_attacked(is_white_king)
    is_check = gs_copy.king_in_check() # Cần hàm này trong GameState
    # Không cần undo gs_copy vì nó là bản sao
    return is_check

# Hàm evaluate_board_pst (Đổi tên để phân biệt)
def evaluate_board_pst(game_state):
    """Trả về đánh giá của bàn cờ dựa trên GameState (Material + PST)"""
    evaluation = 0.0
    for row in range(8):
        for col in range(8):
            piece = game_state.board[row][col]
            square_index = row * 8 + col # Tính index 0-63
            value = get_piece_value_pst(piece, square_index)
            if value is None: value = 0.0
            evaluation += value
    return evaluation


# --- HÀM MÃ HÓA CHO MODEL NNUE ---
# Sao chép từ notebook và điều chỉnh cho GameState.board
pieces_nnue = list('rnbqkpRNBQKP.') # Đảm bảo thứ tự đúng như khi huấn luyện
piece_to_index_nnue = {p: i for i, p in enumerate(pieces_nnue)}
pgn_map_nnue = { # Map từ 'wP', 'bP' sang 'P', 'p'
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k',
    '--': '.' # Map ô trống sang '.'
}

def one_hot_encode_piece_nnue(piece_symbol):
    """Mã hóa one-hot cho một ký tự quân cờ (NNUE)."""
    arr = np.zeros(len(pieces_nnue), dtype=np.float32) # Dùng float32 cho TF
    index = piece_to_index_nnue.get(piece_symbol, piece_to_index_nnue['.']) # Lấy index, mặc định là '.'
    arr[index] = 1.0
    return arr

def encode_board_nnue(board_2d):
    """
    Mã hóa bàn cờ từ list 2D của GameState thành mảng (64, 13) (NNUE).
    Quan trọng: Giả sử board_2d[0][0] là a8, board_2d[7][7] là h1.
    """
    encoded_list = []
    for r in range(8): # Duyệt từ hàng 0 (rank 8) đến hàng 7 (rank 1)
        for c in range(8): # Duyệt từ cột 0 (file a) đến cột 7 (file h)
            pgn_piece = board_2d[r][c]
            piece_symbol = pgn_map_nnue.get(pgn_piece, '.')
            encoded_list.append(one_hot_encode_piece_nnue(piece_symbol))

    # encoded_list là list của 64 mảng (13,)
    # Ghép nối chúng thành mảng (64, 13)
    encoded_array_64_13 = np.array(encoded_list) # Shape sẽ là (64, 13)
    return encoded_array_64_13 # Trả về shape mong đợi


# --- HÀM ĐÁNH GIÁ MỚI SỬ DỤNG NNUE ---
def evaluate_board_nnue(game_state: GameState, loaded_model):
    """
    Hàm đánh giá thế cờ tĩnh sử dụng mô hình NNUE đã huấn luyện.

    Args:
        game_state: Trạng thái hiện tại của trò chơi (từ ChessEngine).
        loaded_model: Mô hình Keras NNUE đã được tải.

    Returns:
        Điểm đánh giá (float), ví dụ: centipawns.
    """
    try:
        # 1. Mã hóa bàn cờ hiện tại sang định dạng NNUE (vector 832,)
        nnue_input_flat = encode_board_nnue(game_state.board)

        # 2. Chuẩn bị đầu vào cho model.predict()
        # Thêm chiều batch: (1, 832)
        nn_input = np.expand_dims(nnue_input_flat, axis=0)

        # 3. Thực hiện dự đoán bằng model NNUE
        # Sử dụng cấu trúc gọi chuẩn của Keras model
        prediction = loaded_model.predict(nn_input, verbose=0)

        # 4. Lấy kết quả điểm số (prediction thường là [[score]])
        score = prediction[0][0]

        # 5. (Quan trọng) Điều chỉnh điểm số nếu cần
        # Nếu NNUE trả về điểm từ góc nhìn của người chơi hiện tại,
        # và Minimax mong đợi điểm từ góc nhìn của Trắng, cần đảo dấu nếu là lượt Đen.
        # Giả sử NNUE trả về điểm từ góc nhìn Trắng (phổ biến).
        # Nếu mô hình trả về điểm từ góc nhìn của người đi hiện tại:
        # if not game_state.white_to_move:
        #     score = -score

        # 6. Trả về điểm số
        return float(score)

    except Exception as e:
        print(f"Lỗi khi đánh giá bằng NNUE: {e}")
        import traceback
        traceback.print_exc()
        # Fallback về hàm đánh giá cũ hoặc trả về 0.0
        print("Fallback về hàm đánh giá PST...")
        return evaluate_board_pst(game_state) # Gọi hàm PST cũ làm fallback
        # Hoặc đơn giản trả về 0.0 nếu không muốn dùng PST nữa
        # return 0.0


# # --- Hàm kiểm tra trạng thái kết thúc (Giữ nguyên) ---
# def check_terminal_state(game_state: GameState):
#     """Kiểm tra trạng thái kết thúc và trả về điểm."""
#     if hasattr(game_state, 'checkmate') and game_state.checkmate:
#         score = -CHECKMATE_SCORE if game_state.white_to_move else CHECKMATE_SCORE
#         return True, score
#     if hasattr(game_state, 'stalemate') and game_state.stalemate:
#         return True, STALEMATE_SCORE
#     # TODO: Thêm kiểm tra hòa khác
#     return False, 0.0

def quiescence_search(game_state: GameState, alpha, beta, ply=0):
    """
    Thực hiện tìm kiếm tĩnh lặng, chỉ xem xét các nước bắt quân.
    Trả về điểm đánh giá từ góc nhìn của Trắng.
    """
    global model_nnue, transposition_table

    # Giới hạn độ sâu QS để tránh tìm kiếm vô hạn (hiếm khi xảy ra với chỉ bắt quân)
    if ply >= MAX_QS_PLY:
        return evaluate_board_nnue(game_state, model_nnue)

    # --- Điểm 1: Đánh giá "Stand-Pat" (Điểm nếu không làm gì thêm) ---
    # Đây là điểm của thế cờ hiện tại, làm mốc so sánh
    try:
        stand_pat_score = evaluate_board_nnue(game_state, model_nnue)
    except Exception as e:
        print(f"Lỗi khi gọi evaluate_board_nnue trong QS (ply {ply}): {e}")
        # Trả về 0.0 hoặc một giá trị an toàn nếu không đánh giá được
        return 0.0

    # --- Điểm 2: Cắt tỉa Alpha-Beta dựa trên điểm Stand-Pat ---
    if game_state.white_to_move: # Lượt Trắng (Maximizer)
        if stand_pat_score >= beta:
            return beta # Điểm này đã đủ tốt, MIN sẽ không chọn nhánh này
        alpha = max(alpha, stand_pat_score)
        best_value_found = stand_pat_score # Giá trị tốt nhất hiện tại là stand-pat
    else: # Lượt Đen (Minimizer)
        if stand_pat_score <= alpha:
            return alpha # Điểm này đã đủ tệ, MAX sẽ không chọn nhánh này
        beta = min(beta, stand_pat_score)
        best_value_found = stand_pat_score # Giá trị tốt nhất hiện tại là stand-pat

    # --- Điểm 3: Tạo và chỉ xem xét các nước bắt quân ---
    try:
        valid_moves = game_state.getValidMoves()
        # Lọc các nước đi là bắt quân
        # Giả sử lớp Move có thuộc tính piece_captured khác '--' khi là nước bắt quân
        capture_moves = [move for move in valid_moves if move.piece_captured != '--']

        # (Tùy chọn nâng cao): Sắp xếp các nước bắt quân (ví dụ: MVV-LVA)
        # capture_moves.sort(key=lambda move: score_capture(move), reverse=True) # Cần hàm score_capture

    except Exception as e:
        print(f"Lỗi khi getValidMoves trong QS (ply {ply}): {e}")
        # Nếu không tạo được nước đi, trả về điểm stand-pat
        return stand_pat_score

    # --- Điểm 4: Duyệt qua các nước bắt quân và gọi đệ quy ---
    for move in capture_moves:
        try:
            game_state.makeMove(move)
            # Gọi đệ quy quiescence_search cho đối phương, tăng ply
            score = quiescence_search(game_state, alpha, beta, ply + 1)
            game_state.undoMove()

            # Cập nhật giá trị tốt nhất và alpha/beta
            if game_state.white_to_move: # Lượt Trắng (Maximizer)
                best_value_found = max(best_value_found, score)
                alpha = max(alpha, best_value_found)
            else: # Lượt Đen (Minimizer)
                best_value_found = min(best_value_found, score)
                beta = min(beta, best_value_found)

            # Cắt tỉa Alpha-Beta trong QS
            if alpha >= beta:
                break # Cắt tỉa

        except Exception as e:
            print(f"Lỗi trong vòng lặp QS (ply {ply}) cho nước đi {move}: {e}")
            try:
                # Cố gắng hoàn tác nếu có lỗi xảy ra sau makeMove
                if game_state.move_log and game_state.move_log[-1] == move:
                     game_state.undoMove()
            except Exception as ue:
                print(f"Lỗi khi undoMove sau lỗi trong QS: {ue}")
            continue # Bỏ qua nước đi lỗi

    # Trả về điểm tốt nhất tìm được (có thể vẫn là stand_pat_score nếu không có nước bắt quân nào cải thiện hoặc bị cắt tỉa)
    return best_value_found

def check_terminal_state(game_state: GameState, ply_from_root=0): # Phiên bản có thể dùng Mate Distance
    CHECKMATE_SCORE = 99999 # Hoặc CHECKMATE_BASE = 30000 nếu dùng Mate Distance
    STALEMATE_SCORE = 0.0
    """Kiểm tra trạng thái kết thúc và trả về điểm."""
    if hasattr(game_state, 'checkmate') and game_state.checkmate:
        # score = -CHECKMATE_SCORE if game_state.white_to_move else CHECKMATE_SCORE # Logic cũ
        # Logic Mate Distance (tùy chọn):
        if not game_state.white_to_move: score = CHECKMATE_SCORE - ply_from_root
        else: score = -CHECKMATE_SCORE + ply_from_root
        return True, score
    if hasattr(game_state, 'stalemate') and game_state.stalemate:
        return True, STALEMATE_SCORE
    return False, 0.0


# --- Hàm lấy độ sâu (Giữ nguyên) ---
def get_depth_based_on_level(ai_level):
    if ai_level == 'easy': return 2
    elif ai_level == 'hard': return 3 # Có thể giảm độ sâu hơn nữa nếu NNUE chậm
    else: return 3 # Mức trung bình


# --- SỬA ĐỔI HÀM MINIMAX ĐỂ DÙNG NNUE EVALUATOR ---
# def minimax(depth, game_state: GameState, alpha, beta, is_maximising_player, ply_from_root=0):
#     """
#     Thuật toán Minimax với cắt tỉa Alpha-Beta.
#     Sử dụng evaluate_board_nnue cho các nút lá.
#     Trả về điểm đánh giá tốt nhất từ góc nhìn của Trắng.
#     """
#     global model_nnue # Cần truy cập model đã tải
#
#     if ply_from_root > MAX_PLY:
#         # Giới hạn độ sâu ply, dùng NNUE để đánh giá
#         return evaluate_board_nnue(game_state, model_nnue)
#
#     is_terminal, terminal_score = check_terminal_state(game_state,ply_from_root )
#     if is_terminal:
#         return terminal_score
#
#     if depth <= 0:
#         # Đạt độ sâu tìm kiếm, dùng hàm đánh giá NNUE
#         # Lưu ý: Việc gọi predict ở mỗi nút lá có thể rất chậm!
#         return quiescence_search(game_state, alpha, beta, ply=0)
#         # return evaluate_board_nnue(game_state, model_nnue)
#
#     try:
#         legal_moves = game_state.getValidMoves()
#         if not legal_moves:
#             return STALEMATE_SCORE
#     except Exception as e:
#         print(f"Error getValidMoves in minimax (depth {depth}): {e}")
#         return 0.0
#
#     # --- XÓA BỎ Sắp xếp nước đi bằng CNN cũ ---
#     # Chúng ta sẽ duyệt qua tất cả các nước đi hợp lệ
#     ordered_moves = legal_moves
#     # Có thể thêm sắp xếp đơn giản ở đây nếu muốn (ví dụ: ưu tiên bắt quân)
#     # Hoặc gọi evaluate_board_nnue cho từng nước đi con (rất chậm) để sắp xếp
#
#     if is_maximising_player: # Lượt Trắng
#         best_value = -float('inf')
#         for move in ordered_moves:
#             try:
#                 game_state.makeMove(move)
#                 value = minimax(depth - 1, game_state, alpha, beta, False, ply_from_root + 1)
#                 game_state.undoMove()
#                 best_value = max(best_value, value)
#                 alpha = max(alpha, best_value)
#                 if alpha >= beta:
#                     break # Cắt tỉa Beta
#             except Exception as e:
#                  print(f"Error in minimax (max) loop, move {move}: {e}")
#                  try: game_state.undoMove()
#                  except Exception as ue: print(f"Undo error after max error: {ue}")
#                  continue
#         return best_value
#     else: # Lượt Đen (AI)
#         best_value = float('inf')
#         for move in ordered_moves:
#             try:
#                 game_state.makeMove(move)
#                 value = minimax(depth - 1, game_state, alpha, beta, True, ply_from_root + 1)
#                 game_state.undoMove()
#                 best_value = min(best_value, value)
#                 beta = min(beta, best_value)
#                 if alpha >= beta:
#                     break # Cắt tỉa Alpha
#             except Exception as e:
#                  print(f"Error in minimax (min) loop, move {move}: {e}")
#                  try: game_state.undoMove()
#                  except Exception as ue: print(f"Undo error after min error: {ue}")
#                  continue
#         return best_value

def minimax(depth, game_state: GameState, alpha, beta, is_maximising_player, ply_from_root=0):
    global model_nnue
    original_alpha = alpha
    original_beta = beta
    # ... (Phần kiểm tra terminal, giới hạn ply, gọi quiescence_search như đã thảo luận) ...

    # --- TT Lookup ---
    # tt_probes += 1 # Đếm số lần tra TT
    current_hash = game_state.get_current_zobrist_hash()  # Sử dụng hàm từ GameState

    if current_hash in transposition_table:
        entry = transposition_table[current_hash]
        # Chỉ sử dụng entry nếu độ sâu tìm kiếm của nó đủ lớn HOẶC LÀ NÚT LÁ TRONG LẦN TÌM KIẾM TRƯỚC
        # Và quan trọng là ply_from_root phải khớp nếu lưu trữ nước đi tốt nhất (không làm ở đây)
        if entry['depth'] >= depth:
            # tt_hits += 1 # Đếm số lần TT hit
            if entry['flag'] == TT_EXACT:
                return entry['score']
            elif entry['flag'] == TT_LOWERBOUND:  # Giá trị lưu là một chặn dưới
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == TT_UPPERBOUND:  # Giá trị lưu là một chặn trên
                beta = min(beta, entry['score'])

            if alpha >= beta:  # Nếu chặn mới gây ra cắt tỉa
                return entry['score']  # Hoặc alpha/beta tùy theo logic bạn muốn

    # Kiểm tra trạng thái kết thúc TRƯỚC KHI KIỂM TRA ĐỘ SÂU
    is_terminal, terminal_score = check_terminal_state(game_state, ply_from_root)
    if is_terminal:
        return terminal_score

    if depth <= 0:
        return quiescence_search(game_state, alpha, beta, ply=0)

    try:
        legal_moves = game_state.getValidMoves()
        if not legal_moves:
            return STALEMATE_SCORE
    except Exception as e:
        print(f"Error getValidMoves in minimax (depth {depth}): {e}")
        return 0.0

    # --- MOVE ORDERING LOGIC ---
    ordered_moves = []

    capture_moves = []
    other_moves_after_captures = []
    for move in legal_moves:
        if move.piece_captured != '--':
            capture_moves.append(move)
        else:
            other_moves_after_captures.append(move)
    capture_moves.sort(key=lambda m: score_move_mvv_lva(m, game_state), reverse=True)
    ordered_moves.extend(capture_moves)

    promotion_moves = []
    other_moves_after_promotions = []
    for move in other_moves_after_captures:
        # Giả sử Move có thuộc tính is_pawn_promotion và promoted_to_piece
        if hasattr(move, 'is_pawn_promotion') and move.is_pawn_promotion:
            promotion_moves.append(move)
        else:
            other_moves_after_promotions.append(move)

    # Sắp xếp ưu tiên phong Hậu
    # Giả sử GameState.white_to_move và piece_simple_value được định nghĩa
    # và Move.promoted_to_piece lưu ký tự quân được phong (vd 'Q', 'R')
    def get_promotion_score(move_obj, gs):
        if not (hasattr(move_obj, 'is_pawn_promotion') and move_obj.is_pawn_promotion):
            return -1  # Không phải phong cấp
        promoted_char = move_obj.promoted_to_piece[1].upper() if hasattr(move_obj,
                                                                         'promoted_to_piece') and move_obj.promoted_to_piece else 'Q'  # Mặc định là Hậu nếu thiếu
        return piece_simple_value.get(promoted_char, 0)

    promotion_moves.sort(key=lambda m: get_promotion_score(m, game_state), reverse=True)
    ordered_moves.extend(promotion_moves)

    # Tạm thời bỏ qua Killer Moves và History Heuristic cho đơn giản
    # Chúng ta sẽ thêm các nước còn lại (chủ yếu là quiet moves)

    # Thêm các nước "yên lặng" còn lại (những nước không phải bắt quân, không phải phong cấp)
    # Bạn có thể muốn sắp xếp chúng thêm bằng is_check_move ở đây nếu cần.
    # Để đơn giản, ta thêm chúng vào cuối.

    # Lấy những nước chưa được thêm vào ordered_moves
    current_ordered_set = set(ordered_moves)  # Dùng set để kiểm tra nhanh
    remaining_quiet_moves = [m for m in other_moves_after_promotions if m not in current_ordered_set]
    ordered_moves.extend(remaining_quiet_moves)

    # --- KẾT THÚC MOVE ORDERING LOGIC (phiên bản đơn giản) ---
    best_move_for_tt = None
    if is_maximising_player:  # Lượt Trắng (Maximizer)
        best_value = -float('inf')
        for move_idx, move in enumerate(ordered_moves):  # Duyệt qua danh sách đã sắp xếp
            try:
                game_state.makeMove(move)
                value = minimax(depth - 1, game_state, alpha, beta, False, ply_from_root + 1)
                game_state.undoMove()

                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    # (Tùy chọn) Cập nhật Killer/History nếu gây cắt tỉa beta
                    # if move.piece_captured == '--': # Chỉ cho non-captures
                    #     update_killer_moves(ply_from_root, move)
                    #     update_history_heuristic(move, depth)
                    break  # Cắt tỉa Beta
            except Exception as e:
                print(f"Error in minimax (max) loop, move {move}: {e}")
                try:
                    # Đảm bảo undoMove được gọi nếu có lỗi sau makeMove
                    # Kiểm tra xem nước đi có trong log trước khi undo để tránh lỗi kép
                    if game_state.move_log and game_state.move_log[-1].moveID == move.moveID:  # Giả sử Move có moveID
                        game_state.undoMove()
                except Exception as ue:
                    print(f"Undo error after max error: {ue}")
                continue  # Bỏ qua nước đi lỗi
        # --- TT Store ---
        entry_data = {'score': best_value,'depth': depth}  # , 'best_move_hash': hash(best_move_for_tt) if best_move_for_tt else None}
        if best_value <= original_alpha:  # Giá trị không cải thiện được alpha -> upper bound
            entry_data['flag'] = TT_UPPERBOUND
        elif best_value >= beta:  # Giá trị gây ra beta cutoff -> lower bound
            entry_data['flag'] = TT_LOWERBOUND
        else:  # Giá trị nằm trong khoảng (alpha, beta) -> exact
            entry_data['flag'] = TT_EXACT
        transposition_table[current_hash] = entry_data
        return best_value
    else:  # Lượt Đen (Minimizer)
        best_value = float('inf')
        for move_idx, move in enumerate(ordered_moves):  # Duyệt qua danh sách đã sắp xếp
            try:
                game_state.makeMove(move)
                value = minimax(depth - 1, game_state, alpha, beta, True, ply_from_root + 1)
                game_state.undoMove()

                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if alpha >= beta:
                    # (Tùy chọn) Cập nhật Killer/History nếu gây cắt tỉa alpha
                    # if move.piece_captured == '--': # Chỉ cho non-captures
                    #     update_killer_moves(ply_from_root, move)
                    #     update_history_heuristic(move, depth)
                    break  # Cắt tỉa Alpha
            except Exception as e:
                print(f"Error in minimax (min) loop, move {move}: {e}")
                try:
                    if game_state.move_log and game_state.move_log[-1].moveID == move.moveID:
                        game_state.undoMove()
                except Exception as ue:
                    print(f"Undo error after min error: {ue}")
                continue  # Bỏ qua nước đi lỗi
        # --- TT Store ---
        entry_data = {'score': best_value,'depth': depth}  # , 'best_move_hash': hash(best_move_for_tt) if best_move_for_tt else None}
        if best_value <= original_alpha:  # Giá trị không cải thiện được alpha -> upper bound
            entry_data['flag'] = TT_UPPERBOUND
        elif best_value >= beta:  # Giá trị gây ra beta cutoff -> lower bound
            entry_data['flag'] = TT_LOWERBOUND
        else:  # Giá trị nằm trong khoảng (alpha, beta) -> exact
            entry_data['flag'] = TT_EXACT
        transposition_table[current_hash] = entry_data
        return best_value


def get_move_from_uci(gs: GameState, uci_string: str) -> Move | None:
    """
    Chuyển đổi chuỗi UCI thành đối tượng Move hợp lệ cho GameState hiện tại.
    Trả về đối tượng Move nếu tìm thấy, None nếu không.
    Hàm này kiểm tra xem nước đi UCI có nằm trong danh sách nước đi hợp lệ không.
    """
    try:
        # Dùng thư viện chess để tạo đối tượng Move từ UCI
        move_from_uci = chess.Move.from_uci(uci_string)

        # Lấy danh sách các đối tượng Move hợp lệ TỪ GameState của bạn
        legal_moves_list = gs.getValidMoves()

        # Tìm đối tượng Move trong legal_moves_list khớp với move_from_uci
        for legal_move in legal_moves_list:

            try:
                start_rank_gs = 7 - legal_move.start_row
                start_file_gs = legal_move.start_col
                end_rank_gs = 7 - legal_move.end_row
                end_file_gs = legal_move.end_col

                start_square_index = chess.square(start_file_gs, start_rank_gs)
                end_square_index = chess.square(end_file_gs, end_rank_gs)

                if start_square_index == move_from_uci.from_square and \
                        end_square_index == move_from_uci.to_square:
                    # Kiểm tra phong cấp (tương tự như trên)
                    if move_from_uci.promotion is not None:
                        if hasattr(legal_move, 'is_pawn_promotion') and legal_move.is_pawn_promotion:
                            expected_promotion_symbol = chess.piece_symbol(move_from_uci.promotion).upper()
                            actual_promotion_symbol = ''
                            # Cần lấy ký tự quân phong cấp từ đối tượng Move của bạn
                            # Ví dụ nếu lưu là 'wQ', 'bN' trong promoted_to_piece
                            if hasattr(legal_move, 'promoted_to_piece') and legal_move.promoted_to_piece:
                                actual_promotion_symbol = legal_move.promoted_to_piece[1].upper()

                            if expected_promotion_symbol == actual_promotion_symbol:
                                return legal_move
                        else:
                            continue
                    else:
                        # Không phải phong cấp
                        return legal_move
            except Exception as coord_err:
                print(f"Lỗi chuyển đổi tọa độ cho move {legal_move}: {coord_err}")
                continue  # Bỏ qua nếu lỗi chuyển đổi

        # Nếu duyệt hết mà không tìm thấy nước đi hợp lệ nào khớp
        print(f"Cảnh báo: Nước đi UCI '{uci_string}' từ sách không phải là nước đi hợp lệ trong thế cờ hiện tại.")
        return None
    except ValueError:
        print(f"Lỗi: Chuỗi UCI không hợp lệ từ sách: '{uci_string}'")
        return None
    except Exception as e:
        print(f"Lỗi không xác định trong get_move_from_uci cho '{uci_string}': {e}")
        return None

# --- SỬA ĐỔI HÀM ROOT ĐỂ KHÔNG DÙNG CNN CŨ ---
def minimax_root(game_state: GameState, ai_level, return_queue, is_maximising_player):
    """
    Tìm nước đi tốt nhất cho AI tại gốc cây tìm kiếm.
    Sử dụng Minimax với hàm đánh giá NNUE cho tìm kiếm.
    """
    global model_nnue, transposition_table # Sử dụng model NNUE đã tải
    transposition_table = {}
    depth = get_depth_based_on_level(ai_level)
    if depth is None:
        print(f"Error: Invalid AI level '{ai_level}'.")
        return_queue.put(None)
        return

    print(f"AI (level {ai_level}, depth {depth}) thinking using NNUE evaluator...")

    # --- KIỂM TRA SÁCH KHAI CUỘC ---
    book_move_obj = None  # Reset biến nước đi từ sách
    print(
        f"[Book Check] opening_book_loaded: {bool(opening_book_loaded)}, len(move_log): {len(game_state.move_log)}, MAX_OPENING_MOVES: {MAX_OPENING_MOVES_IN_AI}")

    if opening_book_loaded and len(game_state.move_log) < MAX_OPENING_MOVES_IN_AI:
        print(f"[Book Check] Attempting book lookup...")
        try:
            print(f"[Book Check] INSIDE TRY block, before calling get_board_fen_position. GameState object: {game_state}") #2
            # Lấy FEN chỉ vị trí quân từ GameState hiện tại
            # Đảm bảo hàm này tồn tại và trả về đúng định dạng FEN vị trí
            print("[Book Check] ABOUT TO CALL get_board_fen_position()")  # THÊM DÒNG NÀY
            current_fen_pos = game_state.get_board_fen_position()
            print(f"[Book Check] RETURNED FROM get_board_fen_position(). FEN is: '{current_fen_pos}'")  # THÊM DÒNG NÀY

            if current_fen_pos in opening_book_loaded:
                candidate_moves_with_counts = opening_book_loaded[current_fen_pos]  # List of ["move_uci", count]

                if candidate_moves_with_counts:
                    print(f"  Tìm thấy thế cờ trong sách. Các lựa chọn: {candidate_moves_with_counts}")

                    # --- CHIẾN LƯỢC CHỌN NƯỚC ĐI TỪ SÁCH ---

                    # Cách 1: Chọn ngẫu nhiên một trong các nước có trong sách
                    # selected_candidate = random.choice(candidate_moves_with_counts)
                    # selected_uci_move = selected_candidate[0]

                    # Cách 2: Chọn nước đi có tần suất cao nhất (ít ngẫu nhiên hơn)
                    # selected_uci_move = candidate_moves_with_counts[0][0] # Vì đã sắp xếp khi tạo sách

                    # Cách 3: Chọn ngẫu nhiên có trọng số dựa trên tần suất
                    moves_uci_list = [item[0] for item in candidate_moves_with_counts]
                    counts = [item[1] for item in candidate_moves_with_counts]
                    if not moves_uci_list:  # Kiểm tra nếu list rỗng sau khi lọc
                        print("  Không có nước đi hợp lệ nào trong sách cho thế cờ này sau khi lọc.")
                    else:
                        total_count = sum(counts)
                        probabilities = [c / total_count for c in counts] if total_count > 0 else None

                        # Cố gắng chọn nước đi và kiểm tra tính hợp lệ
                        attempts = 0
                        max_attempts = len(moves_uci_list)  # Thử tối đa số nước có trong sách

                        while attempts < max_attempts:
                            if probabilities:
                                selected_uci_move = random.choices(moves_uci_list, weights=probabilities, k=1)[0]
                            else:  # Nếu total_count = 0 hoặc chỉ có 1 nước
                                selected_uci_move = random.choice(moves_uci_list)

                            print(f"  Thử chọn nước từ sách (có trọng số): {selected_uci_move}")

                            # Chuyển đổi và quan trọng là KIỂM TRA TÍNH HỢP LỆ
                            potential_book_move = get_move_from_uci(game_state, selected_uci_move)

                            if potential_book_move:
                                book_move_obj = potential_book_move
                                print(f"  AI chọn nước đi hợp lệ từ sách: {selected_uci_move}")
                                break  # Đã tìm thấy nước hợp lệ
                            else:
                                print(f"  Nước đi {selected_uci_move} từ sách không hợp lệ, thử lại...")
                                # Xóa nước đi không hợp lệ khỏi danh sách để không thử lại
                                try:
                                    idx_to_remove = moves_uci_list.index(selected_uci_move)
                                    del moves_uci_list[idx_to_remove]
                                    del counts[idx_to_remove]
                                    if not moves_uci_list: break  # Hết lựa chọn
                                    # Cập nhật lại xác suất nếu cần
                                    total_count = sum(counts)
                                    probabilities = [c / total_count for c in counts] if total_count > 0 else None
                                except ValueError:
                                    pass  # Nếu nước đi đã bị xóa ở lần thử trước
                            attempts += 1

                        if not book_move_obj:
                            print("  Không tìm thấy nước đi hợp lệ nào từ sách sau khi thử.")

        except Exception as book_lookup_error:
            print(f"Lỗi khi tra cứu sách khai cuộc: {book_lookup_error}")
            print(f"Lỗi: {book_lookup_error}")
            import traceback
            traceback.print_exc()
            # Không làm gì cả, sẽ chuyển sang Minimax

    # Nếu tìm được nước đi hợp lệ từ sách, trả về ngay
    if book_move_obj:
        return_queue.put(book_move_obj)
        return

    # --- Lấy trực tiếp các nước đi hợp lệ ---
    try:
        legal_moves = game_state.getValidMoves()
        if not legal_moves:
             print("Warning: No valid moves at root. Cannot make a move.")
             return_queue.put(None)
             return
        # Có thể thêm sắp xếp đơn giản ở đây (VD: ưu tiên bắt quân) nếu muốn
        ordered_moves = legal_moves # Hiện tại không sắp xếp

    except Exception as e:
        print(f"Critical error during getValidMoves at root: {e}")
        return_queue.put(None) # Không thể tìm nước đi
        return

    # --- Thực hiện Minimax trên tất cả các nước đi hợp lệ ---
    best_move_score = -float('inf') if is_maximising_player else float('inf')
    best_move_found = None
    alpha = -float('inf')
    beta = float('inf')

    print(f"Starting Minimax search over {len(ordered_moves)} moves...")

    for i, move in enumerate(ordered_moves):
        try:
            game_state.makeMove(move)
            # Gọi minimax với độ sâu còn lại (depth - 1) và lượt chơi đối phương
            # Minimax giờ sẽ dùng evaluate_board_nnue ở các nút lá
            value = minimax(depth - 1, game_state, alpha, beta, not is_maximising_player, ply_from_root=1)
            game_state.undoMove()
            # print(f"  Root Move {i+1}: {move} -> Score: {value:.2f}") # Log điểm

            # Cập nhật nước đi tốt nhất dựa trên lượt chơi
            if is_maximising_player:
                if value > best_move_score:
                    best_move_score = value
                    best_move_found = move
                alpha = max(alpha, value) # Cập nhật alpha cho root
            else: # Minimising Player (AI Đen)
                if value < best_move_score:
                    best_move_score = value
                    best_move_found = move
                beta = min(beta, value) # Cập nhật beta cho root

            # Không cần cắt tỉa alpha-beta ở root vì ta muốn đánh giá hết các nước đi cấp 1

        except Exception as e:
            print(f"Error during minimax call from root for move {move}: {e}")
            try: game_state.undoMove()
            except Exception as ue: print(f"Undo error after root minimax error: {ue}")
            # Bỏ qua nước đi này

    # Đảm bảo luôn chọn được nước đi nếu có thể
    if best_move_found is None and ordered_moves:
        print("Warning: No best move selected after search, choosing the first legal move.")
        best_move_found = ordered_moves[0]
    elif best_move_found is None:
        print("CRITICAL WARNING: No best move found and no legal moves available!")

    print(f"AI finished thinking. Best Move: {best_move_found} (Score: {best_move_score:.2f})")
    return_queue.put(best_move_found)


# --- Hàm chọn nước đi ngẫu nhiên (Giữ nguyên) ---
def findRandomMove(valid_moves):
     """Chọn một nước đi ngẫu nhiên từ danh sách các nước đi hợp lệ."""
     if valid_moves:
         return random.choice(valid_moves)
     return None

# --- Xóa bỏ các hàm không dùng nữa ---
# del find_best_moves # Hàm dùng CNN cũ
# del board_gamestate_to_cnn_array # Hàm encode cho CNN cũ
# del predict_board_state_cnn # Hàm predict cho CNN cũ


# --- Hàm phụ trợ sắp xếp nước đi ở gốc ---
def order_root_moves(gs: GameState, legal_moves, pv_move):
    """Sắp xếp các nước đi ở gốc, ưu tiên PV move."""
    ordered_moves = []

    # 1. Ưu tiên PV move (nếu có và hợp lệ)
    if pv_move and pv_move in legal_moves:
        ordered_moves.append(pv_move)
        # Tạo danh sách các nước còn lại
        remaining_moves = [m for m in legal_moves if m != pv_move]
    else:
        remaining_moves = list(legal_moves)  # Nếu không có pv_move, bắt đầu với tất cả

    # 2. Sắp xếp các nước còn lại (ví dụ: bắt quân trước)
    # Bạn có thể áp dụng logic phức tạp hơn ở đây (MVV-LVA, v.v.)
    # Ví dụ đơn giản: Bắt quân trước, rồi đến các nước khác
    captures = [m for m in remaining_moves if m.piece_captured != '--']
    non_captures = [m for m in remaining_moves if m.piece_captured == '--']

    # Có thể sắp xếp captures thêm bằng MVV-LVA
    captures.sort(key=lambda m: score_move_mvv_lva(m, gs), reverse=True)  # Cần truy cập game_state ở đây?
    # Hoặc hàm score_move chỉ cần move?
    # Nếu cần gs, phải truyền vào order_root_moves

    ordered_moves.extend(captures)
    ordered_moves.extend(non_captures)  # Các nước còn lại có thể sắp xếp thêm (promotions, killers, history...)

    return ordered_moves

# --- HÀM TÌM NƯỚC ĐI MỚI SỬ DỤNG IDDFS ---
# Đổi tên từ minimax_root hoặc tạo hàm mới
def find_best_move_iddfs(game_state: GameState, ai_level, return_queue, is_maximising_player, time_limit_seconds=10.0):
    """
    Tìm nước đi tốt nhất cho AI sử dụng IDDFS, quản lý thời gian và PV move ordering.
    """
    global model_nnue, transposition_table

    start_time = time.time()
    transposition_table = {}  # Reset TT cho mỗi lần tìm kiếm gốc mới
    print(f"--- IDDFS Search Started ---")
    print(f"Time limit: {time_limit_seconds}s")

    max_depth = get_depth_based_on_level(ai_level)  # Xác định độ sâu tối đa dựa trên level
    if max_depth is None:
        print(f"Error: Invalid AI level '{ai_level}'.")
        return_queue.put(None)
        return

    print(f"AI (level {ai_level}, max_depth {max_depth}) thinking using NNUE evaluator...")

    # --- KIỂM TRA SÁCH KHAI CUỘC ---
    book_move_obj = None  # Reset biến nước đi từ sách
    print(
        f"[Book Check] opening_book_loaded: {bool(opening_book_loaded)}, len(move_log): {len(game_state.move_log)}, MAX_OPENING_MOVES: {MAX_OPENING_MOVES_IN_AI}")

    if opening_book_loaded and len(game_state.move_log) < MAX_OPENING_MOVES_IN_AI:
        print(f"[Book Check] Attempting book lookup...")
        try:
            print(f"[Book Check] INSIDE TRY block, before calling get_board_fen_position. GameState object: {game_state}") #2
            # Lấy FEN chỉ vị trí quân từ GameState hiện tại
            # Đảm bảo hàm này tồn tại và trả về đúng định dạng FEN vị trí
            print("[Book Check] ABOUT TO CALL get_board_fen_position()")  # THÊM DÒNG NÀY
            current_fen_pos = game_state.get_board_fen_position()
            print(f"[Book Check] RETURNED FROM get_board_fen_position(). FEN is: '{current_fen_pos}'")  # THÊM DÒNG NÀY

            if current_fen_pos in opening_book_loaded:
                candidate_moves_with_counts = opening_book_loaded[current_fen_pos]  # List of ["move_uci", count]

                if candidate_moves_with_counts:
                    print(f"  Tìm thấy thế cờ trong sách. Các lựa chọn: {candidate_moves_with_counts}")

                    # --- CHIẾN LƯỢC CHỌN NƯỚC ĐI TỪ SÁCH ---

                    # Cách 1: Chọn ngẫu nhiên một trong các nước có trong sách
                    # selected_candidate = random.choice(candidate_moves_with_counts)
                    # selected_uci_move = selected_candidate[0]

                    # Cách 2: Chọn nước đi có tần suất cao nhất (ít ngẫu nhiên hơn)
                    # selected_uci_move = candidate_moves_with_counts[0][0] # Vì đã sắp xếp khi tạo sách

                    # Cách 3: Chọn ngẫu nhiên có trọng số dựa trên tần suất
                    moves_uci_list = [item[0] for item in candidate_moves_with_counts]
                    counts = [item[1] for item in candidate_moves_with_counts]
                    if not moves_uci_list:  # Kiểm tra nếu list rỗng sau khi lọc
                        print("  Không có nước đi hợp lệ nào trong sách cho thế cờ này sau khi lọc.")
                    else:
                        total_count = sum(counts)
                        probabilities = [c / total_count for c in counts] if total_count > 0 else None

                        # Cố gắng chọn nước đi và kiểm tra tính hợp lệ
                        attempts = 0
                        max_attempts = len(moves_uci_list)  # Thử tối đa số nước có trong sách

                        while attempts < max_attempts:
                            if probabilities:
                                selected_uci_move = random.choices(moves_uci_list, weights=probabilities, k=1)[0]
                            else:  # Nếu total_count = 0 hoặc chỉ có 1 nước
                                selected_uci_move = random.choice(moves_uci_list)

                            print(f"  Thử chọn nước từ sách (có trọng số): {selected_uci_move}")

                            # Chuyển đổi và quan trọng là KIỂM TRA TÍNH HỢP LỆ
                            potential_book_move = get_move_from_uci(game_state, selected_uci_move)

                            if potential_book_move:
                                book_move_obj = potential_book_move
                                print(f"  AI chọn nước đi hợp lệ từ sách: {selected_uci_move}")
                                break  # Đã tìm thấy nước hợp lệ
                            else:
                                print(f"  Nước đi {selected_uci_move} từ sách không hợp lệ, thử lại...")
                                # Xóa nước đi không hợp lệ khỏi danh sách để không thử lại
                                try:
                                    idx_to_remove = moves_uci_list.index(selected_uci_move)
                                    del moves_uci_list[idx_to_remove]
                                    del counts[idx_to_remove]
                                    if not moves_uci_list: break  # Hết lựa chọn
                                    # Cập nhật lại xác suất nếu cần
                                    total_count = sum(counts)
                                    probabilities = [c / total_count for c in counts] if total_count > 0 else None
                                except ValueError:
                                    pass  # Nếu nước đi đã bị xóa ở lần thử trước
                            attempts += 1

                        if not book_move_obj:
                            print("  Không tìm thấy nước đi hợp lệ nào từ sách sau khi thử.")

        except Exception as book_lookup_error:
            print(f"Lỗi khi tra cứu sách khai cuộc: {book_lookup_error}")
            print(f"Lỗi: {book_lookup_error}")
            import traceback
            traceback.print_exc()
            # Không làm gì cả, sẽ chuyển sang Minimax

    # Nếu tìm được nước đi hợp lệ từ sách, trả về ngay
    if book_move_obj:
        return_queue.put(book_move_obj)
        return

    overall_best_move = None
    best_score_so_far = -float('inf') if is_maximising_player else float('inf')
    pv_move_for_next_iter = None  # Nước đi tốt nhất từ lần lặp trước

    # Lấy danh sách nước đi hợp lệ một lần ở gốc
    try:
        initial_legal_moves = game_state.getValidMoves()
        if not initial_legal_moves:
            print("Warning: No valid moves at root. Cannot make a move.")
            return_queue.put(None)
            return
    except Exception as e:
        print(f"Critical error during initial getValidMoves at root: {e}")
        return_queue.put(None)
        return
    # --- VÒNG LẶP IDDFS ---
    try:
        for current_depth in range(1, max_depth + 1):
            print(f"\n--- IDDFS: Starting Depth {current_depth} ---")
            iter_start_time = time.time()  # Thời gian bắt đầu của lần lặp này

            # Sắp xếp các nước đi ở gốc, ưu tiên PV move từ lần lặp trước
            # Cần truyền game_state vào nếu score_move_mvv_lva cần
            # ordered_root_moves = order_root_moves(initial_legal_moves, pv_move_for_next_iter, game_state)
            ordered_root_moves = order_root_moves(game_state, initial_legal_moves,
                                                  pv_move_for_next_iter)  # Giả sử score_move_mvv_lva chỉ cần move

            best_move_this_depth = None
            current_best_score_this_depth = -float('inf') if is_maximising_player else float('inf')
            alpha = -float('inf')
            beta = float('inf')

            # Duyệt qua các nước đi ở gốc cho độ sâu hiện tại
            for i, move in enumerate(ordered_root_moves):
                # Kiểm tra thời gian trước khi thực hiện nước đi tiếp theo ở gốc
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_limit_seconds * 0.95:  # Để lại một chút an toàn
                    print(f"Time limit approaching during depth {current_depth} root move {i + 1}. Stopping search.")
                    raise TimeoutError("Time limit reached")  # Dùng exception để thoát vòng lặp

                try:
                    game_state.makeMove(move)
                    # Gọi hàm minimax đệ quy với độ sâu (current_depth - 1)
                    value = minimax(current_depth - 1, game_state, alpha, beta, not is_maximising_player,
                                    ply_from_root=1)
                    game_state.undoMove()
                    # print(f"  Root Move {i+1}/{len(ordered_root_moves)}: {move} -> Score: {value:.2f}") # Log điểm (có thể rất nhiều)

                    # Cập nhật điểm tốt nhất và alpha/beta cho lần lặp ĐỘ SÂU này
                    if is_maximising_player:
                        if value > current_best_score_this_depth:
                            current_best_score_this_depth = value
                            best_move_this_depth = move
                        alpha = max(alpha, value)  # Alpha-beta pruning ở gốc của mỗi lần lặp
                    else:  # Minimising player
                        if value < current_best_score_this_depth:
                            current_best_score_this_depth = value
                            best_move_this_depth = move
                        beta = min(beta, value)

                    # Không cần cắt tỉa alpha >= beta ở mức gốc này vì chúng ta muốn đánh giá hết các nước đi gốc (trừ khi thời gian hết)

                except Exception as e:
                    print(f"Error during minimax call from root for move {move} at depth {current_depth}: {e}")
                    try:
                        # Đảm bảo undo nếu lỗi sau makeMove
                        if game_state.move_log and game_state.move_log[-1] == move:
                            game_state.undoMove()
                    except Exception as ue:
                        print(f"Undo error after root minimax error: {ue}")
                    # Bỏ qua nước đi này và tiếp tục với nước khác ở gốc

            # --- Kết thúc duyệt các nước ở gốc cho độ sâu hiện tại ---
            iter_elapsed_time = time.time() - iter_start_time

            # Chỉ cập nhật kết quả tổng thể NẾU tìm kiếm ở độ sâu này không bị lỗi và tìm được nước đi
            if best_move_this_depth is not None:
                overall_best_move = best_move_this_depth
                best_score_so_far = current_best_score_this_depth
                pv_move_for_next_iter = best_move_this_depth  # Lưu PV move cho lần lặp sau
                print(
                    f"IDDFS Depth {current_depth} completed in {iter_elapsed_time:.2f}s. Best Move: {overall_best_move} (Score: {best_score_so_far:.2f})")
            else:
                # Có thể xảy ra nếu tất cả các nhánh đều lỗi hoặc không có nước đi hợp lệ (đã kiểm tra ở trên)
                print(
                    f"Warning: No best move found at depth {current_depth} (possibly due to errors in branches). Keeping previous best move.")
                # Giữ nguyên overall_best_move và pv_move_for_next_iter từ độ sâu trước

            # Kiểm tra thời gian tổng thể sau khi hoàn thành một độ sâu
            total_elapsed_time = time.time() - start_time
            print(f"Total time after depth {current_depth}: {total_elapsed_time:.2f}s")
            if total_elapsed_time >= time_limit_seconds:
                print(f"Time limit reached after completing depth {current_depth}.")
                break  # Thoát vòng lặp IDDFS

    except TimeoutError:  # Bắt lỗi hết giờ đã raise ở trên
        print("Search stopped due to time limit (TimeoutError captured). Using best move from last completed depth.")
    except Exception as e_iddfs:
        print(f"--- Error during IDDFS main loop ---")
        print(f"Error: {e_iddfs}")
        import traceback
        traceback.print_exc()
        # Có thể cần trả về nước đi tốt nhất tìm được cho đến nay

    # --- Hoàn tất IDDFS (hoặc bị dừng) ---
    final_elapsed = time.time() - start_time

    # Xử lý trường hợp không tìm được nước đi nào (ví dụ: hết giờ ngay lập tức)
    if overall_best_move is None and initial_legal_moves:
        print("Warning: No best move found (e.g., time limit before depth 1 completion). Choosing first legal move.")
        overall_best_move = initial_legal_moves[0]
    elif overall_best_move is None:
        print(
            "CRITICAL WARNING: No best move found and no legal moves available (This shouldn't happen if initial check passed)!")
        # Có thể trả về None hoặc raise lỗi tùy logic gọi hàm này xử lý

    print(f"--- IDDFS Finished ---")
    if overall_best_move:
        print(f"AI finished thinking (IDDFS). Best Move: {overall_best_move} (Score: {best_score_so_far:.2f})")
    else:
        print(f"AI finished thinking (IDDFS), but failed to find a best move.")
    print(f"Total search time: {final_elapsed:.2f}s")

    return_queue.put(overall_best_move)
