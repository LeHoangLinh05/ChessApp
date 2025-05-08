# Chess_AI.py (Sửa đổi theo hướng Policy Network thô sơ + Giữ lại Minimax truyền thống)

from collections import OrderedDict
from operator import itemgetter
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from ChessEngine import * # Đảm bảo bạn import đúng GameState và Move

# --- Giữ lại các hằng số và bảng PST ---
WEIGHT_MATERIAL_PST = 1
# Các trọng số khác có thể không cần nếu evaluate_board chỉ dùng material+pst
CHECKMATE_SCORE = 99999
STALEMATE_SCORE = 0.0
MAX_PLY = 60 # Giới hạn độ sâu ply để tránh vòng lặp vô hạn

# Đường dẫn tới mô hình CNN đã lưu
path_to_model = 'D:/AI/Chess/ChessApp/Chess/model/morphy_cnn' # <-- THAY ĐỔI THÀNH ĐƯỜNG DẪN CNN CỦA BẠN

# --- Tải Mô hình CNN và kiểm tra Signature ---
global model_cnn
global infer_cnn
global cnn_input_name # Lưu tên input của CNN

try:
    model_cnn = tf.saved_model.load(path_to_model)
    print(f"Model CNN đã được tải thành công từ: {path_to_model}")

    # Lấy signature phục vụ mặc định
    infer_cnn = model_cnn.signatures['serving_default']
    print("Signature 'serving_default' đã được lấy.")

    # --- Quan trọng: Kiểm tra signature của CNN ---
    # Giả định CNN chỉ có MỘT input tensor (trạng thái bàn cờ)
    input_keys = list(infer_cnn.structured_input_signature[1].keys())
    if len(input_keys) == 1:
        cnn_input_name = input_keys[0]
        input_details = infer_cnn.structured_input_signature[1][cnn_input_name]
        print(f"  Input Name: {cnn_input_name}")
        print(f"  Expected Input Shape: {input_details.shape}")
        print(f"  Expected Input Dtype: {input_details.dtype}")
        # Kiểm tra shape (thường là (None, 8, 8, C))
        if not (len(input_details.shape) == 4 and input_details.shape[1] == 8 and input_details.shape[2] == 8):
             print("CẢNH BÁO: Shape input của CNN có vẻ không phải (None, 8, 8, C)!")
    else:
        print(f"LỖI: Signature của CNN không như mong đợi. Tìm thấy {len(input_keys)} inputs: {input_keys}. Cần 1 input.")
        raise ValueError("Signature của model CNN không hợp lệ.")

    # Kiểm tra output signature (thường là 1 output sigmoid)
    output_keys = list(infer_cnn.structured_outputs.keys())
    if len(output_keys) == 1:
        output_details = infer_cnn.structured_outputs[output_keys[0]]
        print(f"  Output Name: {output_keys[0]}")
        print(f"  Output Shape: {output_details.shape}")
        print(f"  Output Dtype: {output_details.dtype}")
        if not (len(output_details.shape) == 2 and output_details.shape[1] == 1):
            print("CẢNH BÁO: Shape output của CNN có vẻ không phải (None, 1)!")
    else:
        print(f"LỖI: CNN có {len(output_keys)} outputs: {output_keys}. Cần 1 output.")
        raise ValueError("Signature output của model CNN không hợp lệ.")

except Exception as e:
    print(f"Lỗi nghiêm trọng khi tải model CNN hoặc lấy signature từ '{path_to_model}': {e}")
    raise RuntimeError(f"Không thể tải model CNN từ '{path_to_model}'") from e

# --- Các bảng PST và hàm phụ trợ (Giữ nguyên) ---
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


def square_to_coord(square):
    """Convert square to coordinates
    """
    return {0: (7, 0), 1: (7, 1), 2: (7, 2), 3: (7, 3), 4: (7, 4), 5: (7, 5), 6: (7, 6), 7: (7, 7),
            8: (6, 0), 9: (6, 1), 10: (6, 2), 11: (6, 3), 12: (6, 4), 13: (6, 5), 14: (6, 6), 15: (6, 7),
            16: (5, 0), 17: (5, 1), 18: (5, 2), 19: (5, 3), 20: (5, 4), 21: (5, 5), 22: (5, 6), 23: (5, 7),
            24: (4, 0), 25: (4, 1), 26: (4, 2), 27: (4, 3), 28: (4, 4), 29: (4, 5), 30: (4, 6), 31: (4, 7),
            32: (3, 0), 33: (3, 1), 34: (3, 2), 35: (3, 3), 36: (3, 4), 37: (3, 5), 38: (3, 6), 39: (3, 7),
            40: (2, 0), 41: (2, 1), 42: (2, 2), 43: (2, 3), 44: (2, 4), 45: (2, 5), 46: (2, 6), 47: (2, 7),
            48: (1, 0), 49: (1, 1), 50: (1, 2), 51: (1, 3), 52: (1, 4), 53: (1, 5), 54: (1, 6), 55: (1, 7),
            56: (0, 0), 57: (0, 1), 58: (0, 2), 59: (0, 3), 60: (0, 4), 61: (0, 5), 62: (0, 6), 63: (0, 7)}[square]


def get_depth_based_on_level(ai_level): # Giữ nguyên
    if ai_level == 'easy': return 2
    elif ai_level == 'hard': return 5 # Giảm depth hard lại để tránh quá chậm


def get_piece_value(piece, square): # Giữ nguyên
    # ... (Code hàm get_piece_value đã sửa lỗi trước đó) ...
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


def evaluate_board(game_state): # Giữ nguyên hàm đánh giá tĩnh
    """Trả về đánh giá của bàn cờ dựa trên GameState từ ChessEngine (Material + PST)"""
    evaluation = 0.0
    for row in range(8):
        for col in range(8):
            piece = game_state.board[row][col]
            square_index = row * 8 + col
            value = get_piece_value(piece, square_index)
            if value is None: value = 0.0
            evaluation += value
    return evaluation

def check_terminal_state(game_state: GameState): # Giữ nguyên
    """Kiểm tra trạng thái kết thúc và trả về điểm."""
    # Cần đảm bảo GameState có thuộc tính checkmate/stalemate
    if hasattr(game_state, 'checkmate') and game_state.checkmate:
        score = -CHECKMATE_SCORE if game_state.white_to_move else CHECKMATE_SCORE
        return True, score
    if hasattr(game_state, 'stalemate') and game_state.stalemate:
        return True, STALEMATE_SCORE
    # TODO: Thêm kiểm tra hòa khác (50 nước, lặp 3 lần,...) nếu GameState hỗ trợ
    return False, 0.0

# --- HÀM MỚI: Chuyển đổi GameState sang Input CNN ---
# Định nghĩa ánh xạ quân cờ sang kênh (tương tự notebook)
piece_to_channel = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, # White
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Black
}
NUM_CHANNELS = 12
pgn_map = {
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k',
    '--': None
}

def board_gamestate_to_cnn_array(game_state):
    """Chuyển đổi trạng thái bàn cờ từ GameState thành mảng (8, 8, 12) cho CNN."""
    cnn_array = np.zeros((8, 8, NUM_CHANNELS), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            pgn_piece = game_state.board[r][c] # Lấy từ GameState
            piece_symbol = pgn_map.get(pgn_piece)
            if piece_symbol:
                channel_index = piece_to_channel.get(piece_symbol)
                if channel_index is not None:
                    # Lưu ý: Cần đảm bảo tọa độ (r, c) của GameState
                    # tương ứng đúng với hàng, cột của mảng numpy
                    # Giả sử board[0][0] là a8, board[7][7] là h1 (phổ biến)
                    # Nếu vậy, cần đảo ngược chỉ số hàng: row_idx = 7 - r
                    row_idx = r # Hoặc row_idx = r nếu board[0][0] là a1
                    col_idx = c
                    if 0 <= row_idx < 8 and 0 <= col_idx < 8:
                         cnn_array[row_idx, col_idx, channel_index] = 1.0
                    else:
                        print(f"Cảnh báo: Chỉ số ({row_idx},{col_idx}) ngoài phạm vi từ (r,c)=({r},{c})")
    # TODO (Quan trọng): Thêm các kênh bổ sung nếu CNN của bạn cần
    # Ví dụ: kênh lượt đi, quyền nhập thành,...
    return cnn_array

# --- HÀM MỚI: Gọi dự đoán từ Model CNN ---
def predict_board_state_cnn(cnn_input_array):
    """
    Dự đoán điểm số cho một trạng thái bàn cờ dùng model CNN đã tải.

    Args:
        cnn_input_array (np.ndarray): Mảng NumPy shape (8, 8, C)

    Returns:
        float: Điểm dự đoán (xác suất good_move) hoặc giá trị rất thấp nếu lỗi.
    """
    global model_cnn, infer_cnn, cnn_input_name
    try:
        # 1. Thêm chiều batch và chuyển thành Tensor
        input_tensor = tf.constant(np.expand_dims(cnn_input_array, axis=0), dtype=infer_cnn.structured_input_signature[1][cnn_input_name].dtype)

        # 2. Tạo dictionary input (chỉ có 1 key)
        input_dict = {cnn_input_name: input_tensor}

        # 3. Gọi dự đoán
        predictions_dict = infer_cnn(**input_dict)

        # 4. Lấy kết quả
        output_key = list(predictions_dict.keys())[0]
        raw_predictions = predictions_dict[output_key] # Shape (1, 1)

        # 5. Trả về giá trị float
        return float(raw_predictions.numpy()[0, 0])

    except Exception as e:
        print(f"Lỗi khi dự đoán bằng CNN: {e}")
        import traceback
        traceback.print_exc()
        # Trả về giá trị rất thấp để nước đi này không được ưu tiên khi có lỗi
        return -float('inf')


# --- VIẾT LẠI: find_best_moves sử dụng CNN ---
def find_best_moves(game_state: GameState, model_cnn, proportion=1.0):
    """
    Sử dụng CNN để đánh giá thế cờ KẾT QUẢ của mỗi nước đi hợp lệ,
    sau đó sắp xếp và trả về các nước đi tốt nhất.

    Args:
        game_state: Trạng thái hiện tại.
        model_cnn: Mô hình CNN đã tải.
        proportion: Tỷ lệ nước đi tốt nhất cần trả về (1.0 = tất cả).

    Returns:
        list[Move]: Danh sách các nước đi tốt nhất (đã sắp xếp).
    """
    try:
        legal_moves = game_state.getValidMoves()
    except Exception as e:
        print(f"Lỗi khi getValidMoves trong find_best_moves: {e}")
        return []

    if not legal_moves:
        return []

    move_scores = {}
    print(f"Evaluating {len(legal_moves)} moves using CNN...") # Thêm log

    for i, move in enumerate(legal_moves):
        if not isinstance(move, Move):
             print(f"Warning: Item {i} is not a Move object: {move}. Skipping.")
             continue
        try:
            # 1. Thực hiện nước đi
            game_state.makeMove(move)

            # 2. Tạo input CNN cho thế cờ MỚI
            current_cnn_input = board_gamestate_to_cnn_array(game_state)

            # 3. Dự đoán điểm cho thế cờ mới
            score = predict_board_state_cnn(current_cnn_input)

            # 4. Hoàn tác nước đi
            game_state.undoMove()

            # Lưu điểm số (từ góc nhìn của người chơi hiện tại)
            # Nếu CNN trả về điểm từ góc nhìn Trắng, và đang là lượt Đen, cần đảo dấu?
            # Giả định CNN trả về P(good_move), luôn dương, không cần đảo dấu.
            move_scores[move] = score
            # print(f"  Move {i+1}/{len(legal_moves)}: {move} -> Score: {score:.4f}") # Log chi tiết

        except Exception as e:
            print(f"Error evaluating move {move} with CNN: {e}")
            # Cố gắng hoàn tác nếu lỗi xảy ra sau makeMove
            try:
                 game_state.undoMove()
                 print(f"  Successfully undid move {move} after error.")
            except Exception as undo_e:
                 print(f"  CRITICAL ERROR: Failed to undo move {move} after error: {undo_e}")
                 # Nếu không undo được, trạng thái game_state có thể bị hỏng!
                 # Có thể cần raise lỗi hoặc thoát khỏi hàm
                 return [] # Trả về rỗng để báo lỗi
            move_scores[move] = -float('inf') # Phạt nặng nước đi gây lỗi

    if not move_scores:
        print("Warning: No moves were evaluated successfully. Returning random move.")
        return [random.choice(legal_moves)] if legal_moves else []

    # Sắp xếp các nước đi theo điểm số giảm dần (nước đi tốt nhất lên đầu)
    sorted_moves = sorted(move_scores.keys(), key=lambda m: move_scores[m], reverse=True)

    # Chọn tỷ lệ nước đi tốt nhất
    num_moves_total = len(sorted_moves)
    num_moves_to_return = int(np.ceil(num_moves_total * proportion))
    num_moves_to_return = min(num_moves_to_return, num_moves_total) # Không vượt quá số lượng có
    num_moves_to_return = max(1, num_moves_to_return) if num_moves_total > 0 else 0 # Ít nhất 1 nếu có

    print(f"CNN evaluation finished. Returning top {num_moves_to_return}/{num_moves_total} moves.")
    return sorted_moves[:num_moves_to_return]

# --- Hàm Minimax (Giữ nguyên - sử dụng evaluate_board ở lá) ---
def minimax(depth, game_state: GameState, alpha, beta, is_maximising_player, ply_from_root=0):
    """
    Thuật toán Minimax với cắt tỉa Alpha-Beta.
    Sử dụng evaluate_board cho các nút lá.
    Trả về điểm đánh giá tốt nhất từ góc nhìn của Trắng.
    """
    if ply_from_root > MAX_PLY:
        return evaluate_board(game_state) # Giới hạn độ sâu ply

    is_terminal, terminal_score = check_terminal_state(game_state)
    if is_terminal:
        return terminal_score

    if depth <= 0:
        # Đạt độ sâu tìm kiếm, dùng hàm đánh giá tĩnh truyền thống
        return evaluate_board(game_state)

    try:
        legal_moves = game_state.getValidMoves()
        if not legal_moves: # Trường hợp stalemate/checkmate đã được xử lý ở trên
            # Nếu vẫn vào đây, có thể là lỗi hoặc tình huống đặc biệt
            # print(f"Warning: No legal moves at depth {depth}, ply {ply_from_root}, but not terminal?")
            return STALEMATE_SCORE # Trả về hòa
    except Exception as e:
        print(f"Error getValidMoves in minimax (depth {depth}): {e}")
        return 0.0 # Trả về điểm an toàn

    # TODO: Sắp xếp nước đi ở đây (nếu không chỉ sắp xếp ở root)
    ordered_moves = legal_moves

    if is_maximising_player: # Lượt Trắng
        best_value = -float('inf')
        for move in ordered_moves:
            try:
                game_state.makeMove(move)
                value = minimax(depth - 1, game_state, alpha, beta, False, ply_from_root + 1)
                game_state.undoMove()
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break # Cắt tỉa Beta
            except Exception as e:
                 print(f"Error in minimax (max) loop, move {move}: {e}")
                 try: game_state.undoMove()
                 except Exception as ue: print(f"Undo error after max error: {ue}")
                 continue # Bỏ qua nước lỗi
        return best_value
    else: # Lượt Đen (AI)
        best_value = float('inf')
        for move in ordered_moves:
            try:
                game_state.makeMove(move)
                value = minimax(depth - 1, game_state, alpha, beta, True, ply_from_root + 1)
                game_state.undoMove()
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if alpha >= beta:
                    break # Cắt tỉa Alpha
            except Exception as e:
                 print(f"Error in minimax (min) loop, move {move}: {e}")
                 try: game_state.undoMove()
                 except Exception as ue: print(f"Undo error after min error: {ue}")
                 continue # Bỏ qua nước lỗi
        return best_value

# --- Hàm Root (Sửa đổi để dùng find_best_moves mới) ---
def minimax_root(game_state: GameState, ai_level, return_queue, is_maximising_player):
    """
    Tìm nước đi tốt nhất cho AI tại gốc cây tìm kiếm.
    Sử dụng CNN để sắp xếp/lọc nước đi ban đầu.
    Sử dụng Minimax với hàm đánh giá tĩnh cho tìm kiếm sâu hơn.
    """
    global model_cnn # Sử dụng model CNN đã tải

    depth = get_depth_based_on_level(ai_level)
    if depth is None:
        print(f"Error: Invalid AI level '{ai_level}'.")
        return_queue.put(None)
        return

    print(f"AI (level {ai_level}, depth {depth}) thinking using CNN guidance...")

    # --- Sử dụng find_best_moves mới dựa trên CNN ---
    # Chọn tỷ lệ proportion dựa trên depth hoặc mức độ AI nếu muốn
    move_proportion = 1.0 # Lấy tất cả nước đi đã sắp xếp ban đầu
    # if depth >= 4: move_proportion = 0.75 # Ví dụ: chỉ xét 75% top moves nếu depth cao

    try:
        # Gọi hàm find_best_moves mới
        ordered_initial_moves = find_best_moves(game_state, model_cnn, proportion=move_proportion)

        if not ordered_initial_moves:
             print("Warning: No valid moves found or evaluated at root. Cannot make a move.")
             # Cố gắng lấy tất cả nước đi hợp lệ nếu find_best_moves lỗi
             all_legal = game_state.getValidMoves()
             if all_legal:
                 print("Falling back to a random legal move.")
                 return_queue.put(random.choice(all_legal))
             else:
                 return_queue.put(None)
             return

    except Exception as e:
        print(f"Critical error during initial move selection (find_best_moves): {e}")
        import traceback
        traceback.print_exc()
        # Fallback cực đoan: chọn nước ngẫu nhiên từ tất cả
        all_legal = game_state.getValidMoves()
        if all_legal:
             print("Falling back to a random legal move due to critical error.")
             return_queue.put(random.choice(all_legal))
        else:
             return_queue.put(None)
        return

    # --- Thực hiện Minimax trên các nước đi đã được sắp xếp/lọc ---
    best_move_score = -float('inf') if is_maximising_player else float('inf')
    best_move_found = None
    alpha = -float('inf')
    beta = float('inf')

    print(f"Starting Minimax search over {len(ordered_initial_moves)} prioritized moves...")

    for i, move in enumerate(ordered_initial_moves):
        # if not isinstance(move, Move): continue # Nên được xử lý bởi find_best_moves

        try:
            game_state.makeMove(move)
            # Gọi minimax với độ sâu còn lại (depth - 1) và lượt chơi đối phương
            value = minimax(depth - 1, game_state, alpha, beta, not is_maximising_player, ply_from_root=1)
            game_state.undoMove()
            # print(f"  Root Move {i+1}: {move} -> Score: {value:.2f}") # Log điểm

            # Cập nhật nước đi tốt nhất dựa trên lượt chơi
            if is_maximising_player:
                if value > best_move_score:
                    best_move_score = value
                    best_move_found = move
                alpha = max(alpha, value) # Cập nhật alpha cho root
            else: # Minimising Player (AI Đen của bạn)
                if value < best_move_score:
                    best_move_score = value
                    best_move_found = move
                beta = min(beta, value) # Cập nhật beta cho root

            # Có thể thêm cắt tỉa ở root nếu muốn, nhưng thường không cần thiết
            # vì chúng ta muốn đánh giá tất cả các nước đi ban đầu tốt nhất
            # if alpha >= beta:
            #    print("Root cutoff occurred.")
            #    break

        except Exception as e:
            print(f"Error during minimax call from root for move {move}: {e}")
            try: game_state.undoMove()
            except Exception as ue: print(f"Undo error after root minimax error: {ue}")
            # Bỏ qua nước đi này

    # Đảm bảo luôn chọn được nước đi nếu có thể
    if best_move_found is None and ordered_initial_moves:
        print("Warning: No best move selected after search, choosing the first prioritized move.")
        best_move_found = ordered_initial_moves[0]
    elif best_move_found is None:
        print("CRITICAL WARNING: No best move found and no initial moves available!")

    print(f"AI finished thinking. Best Move: {best_move_found} (Score: {best_move_score:.2f})")
    return_queue.put(best_move_found)


# --- Hàm chọn nước đi ngẫu nhiên (Giữ nguyên) ---
def findRandomMove(valid_moves):
     """Chọn một nước đi ngẫu nhiên từ danh sách các nước đi hợp lệ."""
     if valid_moves:
         return random.choice(valid_moves)
     return None

# --- Xóa bỏ các hàm không cần thiết liên quan đến model cũ ---
# del get_board_features
# del get_move_features
# del get_possible_moves_data
# del predict_keras_exported
# del EXPECTED_MODEL_COLUMNS