from collections import OrderedDict
from operator import itemgetter
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from ChessEngine import *

WEIGHT_MATERIAL_PST = 1  # Trọng số cho điểm vật chất + PST
WEIGHT_MOBILITY = 0     # Tính cơ động thường có trọng số nhỏ hơn
WEIGHT_CENTER_CONTROL = 0
WEIGHT_KING_SAFETY = 0 # An toàn vua khá quan trọng
WEIGHT_PAWN_STRUCTURE = 0
WEIGHT_BISHOP_PAIR = 0   # Thưởng cho việc có cặp tượng
WEIGHT_ROOK_PLACEMENT = 0 # Vị trí của Xe
MAX_QS_DEPTH = 3
CHECKMATE_SCORE = 99999 # Hoặc một số lớn cụ thể dễ debug hơn float('inf')
STALEMATE_SCORE = 0.0
MAX_PLY = 60
# --- Các ô trung tâm ---
CENTER_SQUARES_MAIN = [(3, 3), (3, 4), (4, 3), (4, 4)] # e4, d4, e5, d5
CENTER_SQUARES_EXTENDED = CENTER_SQUARES_MAIN + [(2, 2), (2, 3), (2, 4), (2, 5),
                                             (3, 2), (3, 5), (4, 2), (4, 5),
                                             (5, 2), (5, 3), (5, 4), (5, 5)] # Mở rộng thêm

path_to_model = 'D:/AI/Chess/ChessApp/Chess/model/morphy_cnn'

global model
# model = tf.saved_model.load(path_to_model)
EXPECTED_MODEL_COLUMNS = []
try:
    model = tf.saved_model.load(path_to_model)
    print("Model đã được tải thành công.")
    infer = model.signatures['serving_default']
    input_details = infer.structured_input_signature[1]
    EXPECTED_MODEL_COLUMNS = list(input_details.keys()) # Gán giá trị sau khi tải thành công
    print(f"\nTotal expected features: {len(EXPECTED_MODEL_COLUMNS)}")
    if len(EXPECTED_MODEL_COLUMNS) != 192:
         print(f"CẢNH BÁO: Số lượng đặc trưng mong đợi ({len(EXPECTED_MODEL_COLUMNS)}) không phải 192!")
    # ... (Phần in signature khác giữ nguyên nếu muốn) ...
except Exception as e:
    print(f"Lỗi nghiêm trọng khi tải model hoặc lấy signature từ '{path_to_model}': {e}")
    raise RuntimeError(f"Không thể tải model từ '{path_to_model}'") from e

# --- Định nghĩa cách mã hóa quân cờ ---
piece_to_int = {
    '--': 0, 'wp': 1, 'wN': 2, 'wB': 3, 'wR': 4, 'wQ': 5, 'wK': 6,
    'bp': -1, 'bN': -2, 'bB': -3, 'bR': -4, 'bQ': -5, 'bK': -6
}
piece_to_float = {k: float(v) for k, v in piece_to_int.items()}
def encode_piece(piece_str):
    return piece_to_float.get(piece_str, 0.0)


def calculate_material_pst_score(game_state):
    """Tính tổng điểm vật chất và điểm từ bảng PST."""
    score = 0.0
    for r in range(8):
        for c in range(8):
            piece = game_state.board[r][c]
            square_index = r * 8 + c
            # Sử dụng hàm get_piece_value đã có (đã bao gồm PST)
            value = get_piece_value(piece, square_index)
            if value is None: # Phòng ngừa lỗi
                 value = 0.0
            score += value
    return score

def calculate_mobility_score(game_state):
    """Tính điểm dựa trên sự chênh lệch số nước đi hợp lệ."""
    # Lưu lượt đi hiện tại
    original_turn = game_state.white_to_move

    # Tính số nước đi của Trắng
    game_state.white_to_move = True
    # Tạo một bản sao trạng thái để tránh thay đổi không mong muốn khi getValidMoves
    # Hoặc đảm bảo getValidMoves không có tác dụng phụ lên trạng thái cốt lõi
    # Ở đây giả định getValidMoves là an toàn hoặc chúng ta chấp nhận rủi ro nhỏ
    white_moves = len(game_state.getValidMoves())

    # Tính số nước đi của Đen
    game_state.white_to_move = False
    black_moves = len(game_state.getValidMoves())

    # Khôi phục lượt đi
    game_state.white_to_move = original_turn

    # Điểm là sự chênh lệch mobility (dương nếu Trắng cơ động hơn)
    mobility_diff = white_moves - black_moves
    return mobility_diff

def calculate_center_control_score(game_state):
    """Tính điểm kiểm soát trung tâm."""
    score = 0.0
    for r, c in CENTER_SQUARES_MAIN: # Duyệt qua các ô trung tâm chính
        piece = game_state.board[r][c]
        if piece != '--':
            # Thưởng nếu quân chiếm ô trung tâm
            bonus = 1.0 if piece[1] == 'p' else 1.5 # Tốt chiếm ít điểm hơn quân khác
            score += bonus if piece[0] == 'w' else -bonus

    # Có thể thêm logic kiểm tra quân nào đang *tấn công* ô trung tâm nữa
    # Điều này phức tạp hơn, cần getValidMoves hoặc logic tấn công riêng
    # Tạm thời bỏ qua để đơn giản

    return score

def get_king_safety_penalty(king_pos, board, opponent_color):
    """Tính điểm phạt cho Vua dựa trên các mối đe dọa xung quanh."""
    penalty = 0.0
    king_r, king_c = king_pos

    # 1. Kiểm tra Tốt che chắn
    pawn_shield_bonus = 0
    shield_row_offset = 1 if opponent_color == 'b' else -1 # Hàng trước mặt vua
    shield_row = king_r + shield_row_offset
    if 0 <= shield_row <= 7:
        for dc in [-1, 0, 1]:
            shield_col = king_c + dc
            if 0 <= shield_col <= 7:
                piece = board[shield_row][shield_col]
                # Thưởng nếu có tốt phe mình che chắn
                if piece[0] != opponent_color and piece[1] == 'p':
                    pawn_shield_bonus += 0.5
                # Phạt nhẹ nếu là tốt đối phương gần vua
                elif piece[0] == opponent_color and piece[1] == 'p':
                     penalty += 0.2

    penalty -= pawn_shield_bonus # Giảm điểm phạt nếu có tốt che

    # 2. Kiểm tra cột mở/nửa mở hướng về Vua (logic đơn giản)
    # Phạt nếu có Xe/Hậu đối phương trên cùng cột và không có quân mình chặn
    friendly_color = 'w' if opponent_color == 'b' else 'b'
    for r_check in range(8):
         piece_on_file = board[r_check][king_c]
         if piece_on_file[0] == opponent_color and (piece_on_file[1] == 'R' or piece_on_file[1] == 'Q'):
             # Kiểm tra xem có quân mình chặn giữa không
             blocked = False
             step = 1 if r_check > king_r else -1
             for r_block in range(king_r + step, r_check, step):
                 if board[r_block][king_c] != '--':
                     blocked = True
                     break
             if not blocked:
                 penalty += 1.5 # Phạt nặng nếu Xe/Hậu dòm ngó trực tiếp

    # 3. Có thể thêm kiểm tra các đường chéo mở, số lượng quân địch tấn công khu vực vua,...

    return penalty


def calculate_king_safety_score(game_state):
    """Tính điểm an toàn Vua cho cả hai bên."""
    white_king_penalty = get_king_safety_penalty(game_state.white_king_location, game_state.board, 'b')
    black_king_penalty = get_king_safety_penalty(game_state.black_king_location, game_state.board, 'w')

    # Điểm dương nếu Vua Trắng an toàn hơn (ít điểm phạt hơn)
    return black_king_penalty - white_king_penalty

def calculate_pawn_structure_score(game_state):
    """Tính điểm dựa trên cấu trúc Tốt."""
    white_pawns = []
    black_pawns = []
    for r in range(8):
        for c in range(8):
            piece = game_state.board[r][c]
            if piece == 'wp':
                white_pawns.append((r, c))
            elif piece == 'bp':
                black_pawns.append((r, c))

    score = 0.0

    # Phạt Tốt chồng (Doubled Pawns)
    white_cols = [c for r, c in white_pawns]
    black_cols = [c for r, c in black_pawns]
    score -= (len(white_cols) - len(set(white_cols))) * 0.5 # Mỗi cặp tốt chồng phạt 0.5
    score += (len(black_cols) - len(set(black_cols))) * 0.5 # Phạt tốt chồng Đen -> lợi cho Trắng

    # Phạt Tốt cô lập (Isolated Pawns) - logic đơn giản: kiểm tra cột kế bên
    for r, c in white_pawns:
        isolated = True
        for other_c in white_cols:
            if abs(c - other_c) == 1:
                isolated = False
                break
        if isolated:
            score -= 0.3
    for r, c in black_pawns:
        isolated = True
        for other_c in black_cols:
            if abs(c - other_c) == 1:
                isolated = False
                break
        if isolated:
            score += 0.3

    # Thưởng Tốt thông (Passed Pawns) - logic cơ bản
    # (Cần kiểm tra không có tốt địch ở trước và cột kế bên)
    # ... (Implement logic kiểm tra Tốt thông nếu cần) ...

    return score

def calculate_bishop_pair_score(game_state):
    """Thưởng nếu một bên có cặp Tượng."""
    white_bishops = 0
    black_bishops = 0
    for r in range(8):
        for c in range(8):
            piece = game_state.board[r][c]
            if piece == 'wB':
                white_bishops += 1
            elif piece == 'bB':
                black_bishops += 1
    score = 0.0
    if white_bishops >= 2:
        score += WEIGHT_BISHOP_PAIR
    if black_bishops >= 2:
        score -= WEIGHT_BISHOP_PAIR
    return score

def is_file_open(col, board):
    """Kiểm tra cột có hoàn toàn không có Tốt nào không."""
    for r in range(8):
        piece = board[r][col]
        if piece[1] == 'p':
            return False
    return True

def is_file_semi_open(col, board, friendly_color):
    """Kiểm tra cột không có Tốt phe mình."""
    for r in range(8):
        piece = board[r][col]
        if piece[0] == friendly_color and piece[1] == 'p':
            return False
    return True

def calculate_rook_placement_score(game_state):
    """Tính điểm cho vị trí của Xe."""
    score = 0.0
    for r in range(8):
        for c in range(8):
            piece = game_state.board[r][c]
            if piece == 'wR':
                if is_file_open(c, game_state.board):
                    score += 0.4 # Thưởng Xe cột mở
                elif is_file_semi_open(c, game_state.board, 'w'):
                    score += 0.2 # Thưởng nhẹ Xe cột nửa mở
                if r == 1: # Hàng 7 của Đen
                    score += 0.5 # Thưởng Xe hàng 7
            elif piece == 'bR':
                if is_file_open(c, game_state.board):
                    score -= 0.4
                elif is_file_semi_open(c, game_state.board, 'b'):
                    score -= 0.2
                if r == 6: # Hàng 2 của Trắng
                    score -= 0.5
    return score

# # --- Hàm Đánh giá Tổng thể ---
# def evaluate_board(game_state):
#     """
#     Trả về đánh giá tổng hợp của bàn cờ, kết hợp nhiều yếu tố.
#     Điểm dương lợi cho Trắng, điểm âm lợi cho Đen.
#     """
#     # Kiểm tra các trường hợp kết thúc game trước
#     if game_state.checkmate:
#         # Nếu Trắng bị chiếu hết -> điểm rất thấp, Đen bị chiếu hết -> điểm rất cao
#         return -9999 if game_state.white_to_move else 9999
#     elif game_state.stalemate:
#         return 0.0 # Hòa cờ
#
#     # Tính điểm từ các thành phần
#     material_pst = calculate_material_pst_score(game_state)
#     mobility = calculate_mobility_score(game_state)
#     center_control = calculate_center_control_score(game_state)
#     king_safety = calculate_king_safety_score(game_state)
#     pawn_structure = calculate_pawn_structure_score(game_state)
#     bishop_pair = calculate_bishop_pair_score(game_state)
#     rook_placement = calculate_rook_placement_score(game_state)
#
#     # Kết hợp điểm với trọng số
#     total_evaluation = (material_pst * WEIGHT_MATERIAL_PST +
#                         mobility * WEIGHT_MOBILITY +
#                         center_control * WEIGHT_CENTER_CONTROL +
#                         king_safety * WEIGHT_KING_SAFETY +
#                         pawn_structure * WEIGHT_PAWN_STRUCTURE +
#                         bishop_pair * WEIGHT_BISHOP_PAIR +
#                         rook_placement * WEIGHT_ROOK_PLACEMENT)
#
#     # Quan trọng: Trả về điểm số từ góc nhìn của người chơi hiện tại
#     # Nếu hàm minimax đã được chuẩn hóa để xử lý góc nhìn, chỉ cần trả về total_evaluation
#     # Nếu không, cần nhân với 1 nếu Trắng đi, -1 nếu Đen đi
#     # return total_evaluation * (1 if game_state.white_to_move else -1) # Nếu minimax chưa chuẩn hóa
#
#     # Giả sử minimax đã được chuẩn hóa (trả về điểm từ góc nhìn Trắng)
#     return total_evaluation

def predict_keras_exported(df_eval, imported_model):
    """
    Dự đoán bằng mô hình Keras đã export (sử dụng serving_default).

    Args:
        df_eval (pd.DataFrame): DataFrame chứa dữ liệu cần dự đoán.
        imported_model: Mô hình đã tải bằng tf.saved_model.load.

    Returns:
        np.ndarray: Mảng NumPy chứa xác suất dự đoán cho lớp dương (good_move).
                   Trả về mảng rỗng nếu có lỗi.
    """
    try:
        # Lấy signature phục vụ mặc định
        infer = imported_model.signatures['serving_default']

        # Lấy danh sách tên input từ signature để đảm bảo khớp
        # structured_input_signature[1] chứa dictionary tên input và TensorSpec
        input_details = infer.structured_input_signature[1]
        all_feature_names = list(input_details.keys())

        # Chuẩn bị dictionary đầu vào
        input_dict = {}
        for col_name in all_feature_names:
            if col_name not in df_eval.columns:
                 print(f"Lỗi: Cột '{col_name}' yêu cầu bởi mô hình không có trong DataFrame.")
                 # Có thể raise lỗi hoặc trả về mảng rỗng tùy cách xử lý lỗi mong muốn
                 # raise ValueError(f"Missing input column: {col_name}")
                 return np.array([]) # Trả về mảng rỗng nếu thiếu cột

            data = df_eval[col_name]
            expected_dtype = input_details[col_name].dtype # Lấy kiểu dữ liệu mong đợi từ signature

            # Chuyển đổi dữ liệu Pandas sang Tensor với kiểu dữ liệu chính xác
            try:
                # Xử lý kiểu string (thường là tf.string)
                if expected_dtype == tf.string:
                     tensor_data = tf.constant(data.astype(str).values, dtype=tf.string)
                # Xử lý kiểu số (thường là tf.float32 sau chuẩn hóa/encoding)
                elif expected_dtype == tf.float32:
                     tensor_data = tf.constant(data.astype(np.float32).values, dtype=tf.float32)
                elif expected_dtype == tf.int64:
                     tensor_data = tf.constant(data.astype(np.int64).values, dtype=tf.int64)
                elif expected_dtype == tf.int32:
                     tensor_data = tf.constant(data.astype(np.int32).values, dtype=tf.int32)
                # Thêm các kiểu khác nếu mô hình của bạn sử dụng (ví dụ: bool)
                else:
                    print(f"Cảnh báo: Kiểu dữ liệu mong đợi chưa xử lý {expected_dtype} cho cột {col_name}. Thử chuyển sang float32.")
                    tensor_data = tf.constant(data.astype(np.float32).values, dtype=tf.float32)

            except Exception as e:
                print(f"Lỗi khi chuyển đổi cột '{col_name}' sang tensor với kiểu {expected_dtype}: {e}")
                return np.array([]) # Trả về mảng rỗng nếu lỗi chuyển đổi

            # Đảm bảo tensor có ít nhất 2 chiều (batch_size, 1) nếu signature yêu cầu
            # Kiểm tra shape mong đợi từ signature nếu cần: input_details[col_name].shape
            # Ví dụ: nếu shape mong đợi là (None, 1) và tensor hiện tại là (batch_size,)
            expected_shape = input_details[col_name].shape
            if len(expected_shape) == 2 and expected_shape[1] == 1 and len(tensor_data.shape) == 1:
                 tensor_data = tf.reshape(tensor_data, (-1, 1))
            elif len(expected_shape) != len(tensor_data.shape):
                 # Cố gắng reshape nếu số chiều không khớp hoàn toàn (ví dụ: thêm chiều batch)
                 # Điều này cần cẩn thận, tùy thuộc vào mô hình
                 # print(f"Warning: Shape mismatch for {col_name}. Expected {expected_shape}, got {tensor_data.shape}. Attempting reshape.")
                 # try:
                 #    # Thêm chiều batch nếu cần
                 #    if len(expected_shape) == len(tensor_data.shape) + 1 and expected_shape[0] is None:
                 #       tensor_data = tf.expand_dims(tensor_data, axis=0)
                 #    # Thêm chiều feature nếu cần
                 #    elif len(expected_shape) == len(tensor_data.shape) + 1 and expected_shape[-1] == 1:
                 #       tensor_data = tf.expand_dims(tensor_data, axis=-1)

                 # except Exception as e:
                 #    print(f"Error reshaping {col_name}: {e}")
                 #    return np.array([])
                 pass # Tạm thời bỏ qua reshape phức tạp, kiểm tra thủ công nếu cần

            input_dict[col_name] = tensor_data


        # Thực hiện dự đoán
        predictions_dict = infer(**input_dict)

        # Lấy tensor đầu ra mong muốn
        # structured_outputs chứa dictionary tên output và giá trị (thường là Tensor)
        output_keys = list(predictions_dict.keys())
        if not output_keys:
             print("Lỗi: Mô hình không trả về output nào.")
             return np.array([])

        # Giả định output mong muốn là output đầu tiên, hoặc bạn biết tên lớp output
        # output_key = 'output_layer' # Hoặc tên lớp output bạn đặt
        output_key = output_keys[0]
        raw_predictions = predictions_dict[output_key]

        # Chuyển đổi tensor kết quả thành mảng NumPy
        # Giả định output là xác suất cho lớp dương, shape (batch_size, 1)
        probabilities = raw_predictions.numpy()
        if probabilities.shape[-1] == 1:
            return probabilities[:, 0] # Lấy cột đầu tiên nếu shape là (batch, 1)
        elif len(probabilities.shape) == 1:
             return probabilities # Trả về trực tiếp nếu shape là (batch,)
        else:
             print(f"Cảnh báo: Shape của output không mong đợi: {probabilities.shape}. Trả về cột đầu tiên.")
             return probabilities[:, 0]


    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]) # Trả về mảng rỗng khi có lỗi


def get_board_features(game_state):
    """Trả về các đặc trưng của bàn cờ"""
    board_features = []
    for row in range(8):
        for col in range(8):
            piece = game_state.board[row][col]  # Lấy quân cờ từ bàn cờ của GameState
            board_features.append(piece)
    return board_features

def get_move_features(move):
    """Trả về đặc trưng của một nước đi"""
    from_ = np.zeros(64)
    to_ = np.zeros(64)
    from_[move.start_row * 8 + move.start_col] = 1
    to_[move.end_row * 8 + move.end_col] = 1
    return from_, to_


def get_possible_moves_data(current_board):
    """
    Trả về pd.DataFrame của tất cả các nước đi hợp lệ dùng cho dự đoán,
    với cột trạng thái bàn cờ là chuỗi, cột nước đi là float,
    và các cột được sắp xếp theo EXPECTED_MODEL_COLUMNS.
    """
    # <<< SỬA LỖI 1: Khai báo sử dụng biến toàn cục >>>
    global EXPECTED_MODEL_COLUMNS

    # Kiểm tra lại biến toàn cục sau khi khai báo global
    if not EXPECTED_MODEL_COLUMNS or len(EXPECTED_MODEL_COLUMNS) != 192:
         print("Lỗi nghiêm trọng: EXPECTED_MODEL_COLUMNS không hợp lệ sau khi khai báo global.")
         return pd.DataFrame()

    data = []
    moves = current_board.getValidMoves()
    if not moves:
        return pd.DataFrame(columns=EXPECTED_MODEL_COLUMNS)

    # --- 1. Lấy đặc trưng bàn cờ dạng CHUỖI ---
    board_state_strings = []
    for r in range(8):
        for c in range(8):
            board_state_strings.append(current_board.board[r][c])

    # --- 2. Hàm nội bộ để tính one-hot float32 cho nước đi ---
    def calculate_move_features(move):
        from_square_ = np.zeros(64, dtype=np.float32)
        to_square_ = np.zeros(64, dtype=np.float32)
        if move:
             from_square_[move.start_row * 8 + move.start_col] = 1.0
             to_square_[move.end_row * 8 + move.end_col] = 1.0
        return from_square_, to_square_

    # --- 3. Thu thập dữ liệu ---
    for move in moves:
        from_square, to_square = calculate_move_features(move)
        row_mixed_type = board_state_strings + from_square.tolist() + to_square.tolist()
        data.append(row_mixed_type)

    # --- 4. Tạo DataFrame ban đầu ---
    board_feature_names_temp = [f"{chr(c + 97)}{8 - r}" for r in range(8) for c in range(8)]
    move_from_feature_names_temp = ['from_' + s for s in board_feature_names_temp]
    move_to_feature_names_temp = ['to_' + s for s in board_feature_names_temp]
    columns = board_feature_names_temp + move_from_feature_names_temp + move_to_feature_names_temp
    df = pd.DataFrame(data=data, columns=columns)

    # --- 5. Ép kiểu cột nước đi ---
    try:
         for col in move_from_feature_names_temp + move_to_feature_names_temp:
             if col in df.columns:
                 df[col] = df[col].astype(np.float32)
    except Exception as e:
         print(f"Lỗi khi ép kiểu cột nước đi thành float32: {e}")
         # return pd.DataFrame(columns=EXPECTED_MODEL_COLUMNS) # Bỏ comment nếu muốn dừng khi lỗi

    # --- 6. Sắp xếp lại cột ---
    try:
        df = df[EXPECTED_MODEL_COLUMNS]
    except KeyError as e:
        print(f"Lỗi nghiêm trọng khi sắp xếp cột: Không tìm thấy cột '{e}'.")
        missing_cols = [col for col in EXPECTED_MODEL_COLUMNS if col not in df.columns]
        print(f"Các cột mong đợi bị thiếu: {missing_cols}")
        extra_cols = [col for col in df.columns if col not in EXPECTED_MODEL_COLUMNS]
        print(f"Các cột có trong df nhưng không mong đợi: {extra_cols}")
        return pd.DataFrame(columns=EXPECTED_MODEL_COLUMNS)
    except Exception as ex:
        print(f"Lỗi không xác định khi sắp xếp lại cột: {ex}")
        return pd.DataFrame(columns=EXPECTED_MODEL_COLUMNS)

    return df


def find_best_moves(game_state, model, proportion=1):
    """Trả về danh sách các nước đi tốt nhất (Move) cho một đối tượng GameState."""
    moves = game_state.getValidMoves()
    if not moves:
        return [] # Không có nước đi nào

    df_eval = get_possible_moves_data(game_state)
    if df_eval.empty:
        print("Cảnh báo: Không tạo được dữ liệu để đánh giá.")
        # Trả về nước đi ngẫu nhiên nếu có moves nhưng không tạo được df
        return [random.choice(moves)] if moves else []


    # Gọi hàm dự đoán đã cập nhật
    good_move_probas = predict_keras_exported(df_eval, model)

    # Kiểm tra nếu hàm predict trả về lỗi (mảng rỗng)
    if good_move_probas.size == 0:
         print("Lỗi khi lấy dự đoán từ mô hình. Chọn nước đi ngẫu nhiên.")
         return [random.choice(moves)] # Chọn ngẫu nhiên khi lỗi

    # Kiểm tra số lượng dự đoán có khớp với số nước đi không
    if len(good_move_probas) != len(moves):
        print(f"Lỗi: Số lượng dự đoán ({len(good_move_probas)}) không khớp số nước đi ({len(moves)}). Chọn nước đi ngẫu nhiên.")
        return [random.choice(moves)]


    # Tạo từ điển và sắp xếp
    dict_ = {}
    for move, proba in zip(moves, good_move_probas):
        # Đảm bảo proba là float (từ mảng numpy)
        dict_[move] = float(proba)

    dict_ = OrderedDict(sorted(dict_.items(), key=itemgetter(1), reverse=True))

    best_moves = list(dict_.keys())
    if not best_moves: # Trường hợp hi hữu không có key nào sau khi sort
         print("Cảnh báo: Không có nước đi nào trong danh sách đã sắp xếp. Chọn ngẫu nhiên.")
         return [random.choice(moves)]

    num_moves_to_return = int(np.ceil(len(best_moves) * proportion))
    # Đảm bảo num_moves_to_return không lớn hơn số lượng best_moves có
    num_moves_to_return = min(num_moves_to_return, len(best_moves))
    # Đảm bảo luôn trả về ít nhất 1 nếu có thể
    num_moves_to_return = max(1, num_moves_to_return)

    best_moves_to_return = best_moves[:num_moves_to_return]

    return best_moves_to_return



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

def get_depth_based_on_level(ai_level):
    """Map the AI level to the corresponding depth"""
    if ai_level == 'easy':
        return 2
    elif ai_level == 'hard':
        return 5



# Trong Chess_AI.py

# Giả định các bảng PST (pawn_white_eval, ..., king_black_eval)
# và hàm square_to_coord(square) đã được định nghĩa ở trên

def get_piece_value(piece, square):
    """Trả về giá trị của quân cờ tại ô chỉ định, khớp với ChessEngine.
       Đã sửa lỗi xử lý loại quân chữ thường.

    Args:
        piece (str): Chuỗi đại diện quân cờ ('wp', 'bK', '--', etc.).
        square (int): Chỉ số ô (0-63).

    Returns:
        float: Giá trị của quân cờ.
    """
    if piece == '--':
        return 0.0

    try:
        x, y = square_to_coord(square)
    except KeyError:
        print(f"Lỗi: Chỉ số ô không hợp lệ '{square}' trong square_to_coord.")
        return 0.0

    piece_color = piece[0]
    piece_type = piece[1]
    piece_type_upper = piece_type.upper() # <-- Chuẩn hóa thành chữ HOA

    sign = 1.0 if piece_color == 'w' else -1.0

    base_value = 0.0
    pst_value = 0.0

    try:
        # --- Sử dụng piece_type_upper để so sánh ---
        if piece_type_upper == 'P':
            base_value = 10.0
            pst_value = pawn_white_eval[x][y] if piece_color == 'w' else pawn_black_eval[x][y]
        elif piece_type_upper == 'N':
            base_value = 30.0
            pst_value = knight_white_eval[x][y] if piece_color == 'w' else knight_black_eval[x][y]
        elif piece_type_upper == 'B':
            base_value = 30.0
            pst_value = bishop_white_eval[x][y] if piece_color == 'w' else bishop_black_eval[x][y]
        elif piece_type_upper == 'R':
            base_value = 50.0
            pst_value = rook_white_eval[x][y] if piece_color == 'w' else rook_black_eval[x][y]
        elif piece_type_upper == 'Q':
            base_value = 90.0
            pst_value = queen_white_eval[x][y] if piece_color == 'w' else queen_black_eval[x][y]
        elif piece_type_upper == 'K':
            base_value = 900.0
            pst_value = king_white_eval[x][y] if piece_color == 'w' else king_black_eval[x][y]
        else:
            # Khối else này giờ chỉ được kích hoạt nếu piece[1] không phải là ký tự quân cờ hợp lệ
            print(f"Cảnh báo: Loại quân cờ không hợp lệ '{piece_type}' từ piece '{piece}'")
            return 0.0 # <-- Đảm bảo luôn trả về giá trị số
    except IndexError:
         print(f"Lỗi: Tọa độ ({x}, {y}) nằm ngoài phạm vi bảng PST khi xử lý quân {piece}.")
         return 0.0
    except Exception as e:
         print(f"Lỗi không xác định khi lấy giá trị PST cho {piece} tại ({x},{y}): {e}")
         return 0.0

    final_value = sign * (base_value + pst_value)
    return final_value

# --- Hàm evaluate_board (giữ nguyên như đã sửa trước đó) ---
def evaluate_board(game_state):
    """Trả về đánh giá của bàn cờ dựa trên GameState từ ChessEngine"""
    evaluation = 0.0
    for row in range(8):
        for col in range(8):
            piece = game_state.board[row][col]
            square_index = row * 8 + col # Tính chỉ số ô từ 0-63
            value = get_piece_value(piece, square_index) # Gọi hàm đã sửa
            if value is None: # Thêm kiểm tra phòng ngừa (dù không nên xảy ra nữa)
                 print(f"LỖI NGHIÊM TRỌNG: get_piece_value trả về None cho piece '{piece}' tại ô {square_index}")
                 value = 0.0 # Gán giá trị mặc định để tránh crash
            evaluation += value
    return evaluation

# def quiescence_search(game_state, alpha, beta, is_maximising_player, current_qs_depth):
#     """
#     Thực hiện tìm kiếm tĩnh lặng để đánh giá các vị trí không ổn định.
#     Chỉ xem xét các nước đi bắt quân (hoặc các nước đi "ồn ào" khác).
#     Trả về điểm đánh giá từ góc nhìn của Trắng.
#     """
#
#     # --- 1. Đánh giá "Stand Pat" (Điểm nếu không làm gì) ---
#     # Đây là điểm số tối thiểu (cho min player) hoặc tối đa (cho max player)
#     # mà chúng ta có thể đảm bảo nếu không thực hiện nước đi ồn ào nào.
#     stand_pat_score = evaluate_board(game_state) # Điểm đánh giá tĩnh hiện tại
#
#     # --- 2. Cập nhật Alpha/Beta dựa trên Stand Pat & Kiểm tra Cutoff sớm ---
#     if is_maximising_player:
#         alpha = max(alpha, stand_pat_score) # Max player có thể đạt ít nhất điểm này
#     else:
#         beta = min(beta, stand_pat_score)   # Min player có thể giữ điểm không cao hơn điểm này
#
#     # Nếu stand_pat đã gây ra cutoff, không cần tìm kiếm thêm
#     if alpha >= beta:
#         return stand_pat_score
#
#     # --- 3. Kiểm tra giới hạn độ sâu QS ---
#     if current_qs_depth >= MAX_QS_DEPTH:
#         return stand_pat_score # Đạt giới hạn, trả về đánh giá tĩnh
#
#     # --- 4. Tạo danh sách các nước đi "ồn ào" (chủ yếu là bắt quân) ---
#     noisy_moves = []
#     try:
#         all_moves = game_state.getValidMoves() # Lấy tất cả nước đi hợp lệ
#         for move in all_moves:
#             # Ưu tiên các nước bắt quân
#             if move.is_capture: # Giả sử thuộc tính này tồn tại
#             # Hoặc: if move.piece_captured != '--':
#                 noisy_moves.append(move)
#             # TODO (Tùy chọn): Thêm các nước đi khác được coi là "ồn ào"
#             # Ví dụ: Phong cấp (move.is_pawn_promotion), hoặc thậm chí là Chiếu
#     except Exception as e:
#          print(f"Lỗi khi getValidMoves trong QS: {e}")
#          return stand_pat_score # Trả về đánh giá tĩnh nếu không lấy được nước đi
#
#     # --- 5. Nếu không có nước đi ồn ào, trả về Stand Pat ---
#     if not noisy_moves:
#         return stand_pat_score
#
#     # --- TODO (Tùy chọn nâng cao): Sắp xếp noisy_moves ---
#     # Sắp xếp các nước bắt quân (ví dụ: MVV-LVA - Bắt quân giá trị cao nhất bằng quân giá trị thấp nhất trước)
#     # có thể tăng hiệu quả cắt tỉa alpha-beta đáng kể trong QS.
#
#     # --- 6. Thực hiện vòng lặp Minimax chỉ trên các nước đi ồn ào ---
#     if is_maximising_player:
#         best_value = stand_pat_score # Khởi tạo với điểm stand_pat
#         for move in noisy_moves:
#             try:
#                 game_state.makeMove(move)
#                 # Gọi đệ quy quiescence_search, tăng độ sâu QS
#                 value = quiescence_search(game_state, alpha, beta, False, current_qs_depth + 1)
#                 game_state.undoMove()
#
#                 best_value = max(best_value, value)
#                 alpha = max(alpha, best_value) # Cập nhật alpha
#
#                 # Cắt tỉa Beta
#                 if alpha >= beta:
#                     break # Dừng duyệt các nước đi ồn ào còn lại
#             except Exception as e:
#                  print(f"Lỗi trong QS (max) khi xử lý {move}: {e}")
#                  try: game_state.undoMove() # Cố gắng hoàn tác nếu lỗi
#                  except: pass
#                  continue # Bỏ qua nước đi lỗi này
#         return best_value # Trả về điểm tốt nhất tìm được cho Max player
#     else: # Minimising player (AI Đen của bạn khi is_maximising_player=False)
#         best_value = stand_pat_score # Khởi tạo với điểm stand_pat
#         for move in noisy_moves:
#             try:
#                 game_state.makeMove(move)
#                 # Gọi đệ quy quiescence_search, tăng độ sâu QS
#                 value = quiescence_search(game_state, alpha, beta, True, current_qs_depth + 1)
#                 game_state.undoMove()
#
#                 best_value = min(best_value, value)
#                 beta = min(beta, best_value) # Cập nhật beta
#
#                 # Cắt tỉa Alpha
#                 if alpha >= beta:
#                     break # Dừng duyệt các nước đi ồn ào còn lại
#             except Exception as e:
#                  print(f"Lỗi trong QS (min) khi xử lý {move}: {e}")
#                  try: game_state.undoMove()
#                  except: pass
#                  continue
#         return best_value # Trả về điểm tốt nhất tìm được cho Min player (điểm thấp nhất theo góc nhìn Trắng)

# def minimax(depth, game_state, alpha, beta, is_maximising_player): # Sửa: Nhận depth
#     """Minimax algorithm with alpha-beta pruning"""
#     # Trường hợp cơ sở: độ sâu bằng 0
#     if depth == 0:
#         # Giả sử evaluate_board hoạt động đúng
#         return evaluate_board(game_state)
#
#     # --- Logic lấy nước đi ---
#     # Trong các bước đệ quy, thường xem xét tất cả nước đi hợp lệ
#     # Logic chọn lọc (dùng find_best_moves) thường chỉ áp dụng ở root hoặc độ sâu nông
#     legal_moves = game_state.getValidMoves() # Lấy tất cả nước đi hợp lệ
#
#     # Xử lý trường hợp không còn nước đi (có thể là checkmate/stalemate ở nút lá)
#     if not legal_moves:
#          return evaluate_board(game_state) # Trả về đánh giá tĩnh
#
#     if is_maximising_player:
#         best_value = -9999 # Đổi tên để rõ ràng hơn
#         for move in legal_moves:
#             if not isinstance(move, Move): continue # Kiểm tra lại kiểu dữ liệu
#
#             try:
#                 game_state.makeMove(move)
#                 # CHỈ GỌI ĐỆ QUY Ở ĐÂY
#                 best_value = max(best_value, minimax(depth - 1, game_state, alpha, beta, not is_maximising_player))
#                 game_state.undoMove()
#
#             except Exception as e:
#                 print(f"Lỗi trong minimax (max) khi xử lý {move}: {e}")
#                 # Cân nhắc hoàn tác nếu lỗi xảy ra sau makeMove
#                 try:
#                     game_state.undoMove()
#                 except:
#                     pass
#                 continue  # Bỏ qua nước đi lỗi
#
#             alpha = max(alpha, best_value)
#             if beta <= alpha:  # Beta cut-off
#                 break  # Không cần break, chỉ cần return
#         return best_value  # Trả về giá trị tốt nhất tìm được
#
#     else: # Minimising player
#         best_value = 9999
#         for move in legal_moves:
#
#             try:
#                 game_state.makeMove(move)
#                 # Gọi đệ quy với depth - 1
#                 best_value = min(best_value, minimax(depth - 1, game_state, alpha, beta, not is_maximising_player))
#                 game_state.undoMove()
#
#             except Exception as e:
#                 print(f"Lỗi trong minimax (min) khi xử lý {move}: {e}")
#                 try:
#                     game_state.undoMove()
#                 except:
#                     pass
#                 continue
#
#             beta = min(beta, best_value)
#             if beta <= alpha:  # Alpha cut-off
#                 break
#         return best_value

def minimax(depth, game_state: GameState, alpha, beta, is_maximising_player, ply_from_root=0):
    """
    Thuật toán Minimax với cắt tỉa Alpha-Beta, Tìm kiếm Tĩnh lặng (Quiescence Search),
    và hỗ trợ cơ bản cho Bảng Chuyển vị (Transposition Table).
    Trả về điểm đánh giá tốt nhất từ góc nhìn của Trắng.
    """
    # --- 1. Kiểm tra Giới hạn Ply ---
    if ply_from_root > MAX_PLY:
        return evaluate_board(game_state)

    # --- 2. Transposition Table Lookup ---
    original_alpha = alpha # Lưu lại để xác định flag TT sau này
    # zobrist_key = calculate_zobrist_key(game_state, zobrist_keys) # Lấy Zobrist key
    # tt_entry = transposition_table.get(zobrist_key)

    # if tt_entry and tt_entry['depth'] >= depth:
    #     # Tìm thấy entry trong TT với độ sâu đủ lớn
    #     if tt_entry['flag'] == 'EXACT':
    #         return tt_entry['score'] # Trả về ngay nếu là điểm chính xác
    #     elif tt_entry['flag'] == 'LOWER_BOUND': # Điểm lưu là cận dưới
    #         alpha = max(alpha, tt_entry['score']) # Cập nhật alpha
    #     elif tt_entry['flag'] == 'UPPER_BOUND': # Điểm lưu là cận trên
    #         beta = min(beta, tt_entry['score'])   # Cập nhật beta

    #     # Kiểm tra cắt tỉa ngay sau khi cập nhật alpha/beta từ TT
    #     if alpha >= beta:
    #         return tt_entry['score'] # Cutoff dựa trên thông tin TT

    # --- 3. Xử lý Nút Lá (Kết thúc Game hoặc Độ sâu = 0) ---
    is_terminal, terminal_score = check_terminal_state(game_state)
    if is_terminal:
        # Lưu kết quả terminal vào TT nếu muốn (độ sâu rất lớn)
        # store_in_tt(zobrist_key, 99, terminal_score, 'EXACT', None) # depth=99 ví dụ
        return terminal_score

    if depth <= 0:
        # Gọi Quiescence Search khi hết độ sâu tìm kiếm chính
        # QS cũng nên có cơ chế tra cứu/lưu TT riêng hoặc dùng chung TT chính
        # Lưu kết quả QS vào TT (với depth=0 hoặc độ sâu QS thực tế)
        # store_in_tt(zobrist_key, 0, qs_score, 'EXACT', None) # Giả sử QS trả về điểm EXACT
        return evaluate_board(game_state)

    # --- 4. Tạo và Sắp xếp Nước đi ---
    try:
        legal_moves = game_state.getValidMoves()
        if not legal_moves:
            # Điều này chỉ xảy ra nếu check_terminal_state sai hoặc có lỗi khác
            print(f"Cảnh báo: Không có nước đi tại depth={depth}, ply={ply_from_root}")
            # Trả về điểm hòa hoặc đánh giá tĩnh tùy tình huống
            return STALEMATE_SCORE # Hoặc evaluate_board(game_state)
    except Exception as e:
        print(f"Lỗi getValidMoves tại depth={depth}, ply={ply_from_root}: {e}")
        # Trả về giá trị an toàn khi có lỗi
        return 0.0

    # --- Sắp xếp nước đi (QUAN TRỌNG!) ---
    # ordered_moves = sort_moves(legal_moves, game_state, tt_entry) # Hàm sắp xếp riêng
    ordered_moves = legal_moves # Tạm thời chưa sắp xếp

    # --- 5. Vòng lặp Minimax ---
    best_move_for_tt = None # Nước đi tốt nhất tìm được ở nút này để lưu vào TT

    if is_maximising_player: # Lượt Trắng (Tối đa hóa)
        best_value = -float('inf')
        for move in ordered_moves:
            # if not isinstance(move, Move): continue # Thừa nếu getValidMoves đúng

            try:
                game_state.makeMove(move)
                # zobrist_key = update_zobrist_key(zobrist_key, move, game_state, zobrist_keys) # Cập nhật key
                value = minimax(depth - 1, game_state, alpha, beta, False, ply_from_root + 1)
                game_state.undoMove()
                # zobrist_key = calculate_zobrist_key(game_state, zobrist_keys) # Tính lại key sau undo (hoặc update ngược)

                # Cập nhật giá trị tốt nhất
                if value > best_value:
                    best_value = value
                    best_move_for_tt = move # Lưu nước đi dẫn đến giá trị tốt nhất

                # Cập nhật alpha
                alpha = max(alpha, best_value)

                # Cắt tỉa Beta
                if alpha >= beta:
                    # TODO: Có thể thêm logic lưu nước đi gây cắt tỉa (beta-cutoff move)
                    # vào các cấu trúc như Killer Moves hoặc History Heuristic để cải thiện sắp xếp
                    break # Dừng duyệt các nước còn lại

            except Exception as e:
                 print(f"Lỗi trong minimax (max) loop, move {move}: {e}")
                 try: game_state.undoMove()
                 except Exception as undo_e: print(f"Lỗi undo sau lỗi max: {undo_e}")
                 # Xem xét nên làm gì khi có lỗi, trả về giá trị tệ?
                 # Hoặc bỏ qua nước đi này và tiếp tục? (hiện tại đang bỏ qua)

        # --- Lưu vào Transposition Table ---
        # tt_flag = ''
        # if best_value <= original_alpha: # Không cải thiện được alpha -> Cận trên
        #     tt_flag = 'UPPER_BOUND'
        # elif best_value >= beta: # Bị cắt bởi beta -> Cận dưới
        #     tt_flag = 'LOWER_BOUND'
        # else: # Nằm trong khoảng (alpha, beta) ban đầu -> Chính xác
        #     tt_flag = 'EXACT'
        # store_in_tt(zobrist_key, depth, best_value, tt_flag, best_move_for_tt)

        return best_value

    else: # Lượt Đen (Tối thiểu hóa - AI của bạn)
        best_value = float('inf')
        for move in ordered_moves:
            # if not isinstance(move, Move): continue

            try:
                game_state.makeMove(move)
                # zobrist_key = update_zobrist_key(zobrist_key, move, game_state, zobrist_keys)
                value = minimax(depth - 1, game_state, alpha, beta, True, ply_from_root + 1)
                game_state.undoMove()
                # zobrist_key = calculate_zobrist_key(game_state, zobrist_keys)

                if value < best_value:
                    best_value = value
                    best_move_for_tt = move

                beta = min(beta, best_value)

                if alpha >= beta:
                    # TODO: Logic lưu nước đi gây cắt tỉa (alpha-cutoff move)
                    break

            except Exception as e:
                 print(f"Lỗi trong minimax (min) loop, move {move}: {e}")
                 try: game_state.undoMove()
                 except Exception as undo_e: print(f"Lỗi undo sau lỗi min: {undo_e}")

        # --- Lưu vào Transposition Table ---
        # tt_flag = ''
        # if best_value <= alpha: # Bị cắt bởi alpha -> Cận trên
        #     tt_flag = 'UPPER_BOUND'
        # elif best_value >= beta: # Không cải thiện được beta -> Cận dưới
        #      tt_flag = 'LOWER_BOUND'
        # else: # Nằm trong khoảng (alpha, beta) ban đầu -> Chính xác
        #      tt_flag = 'EXACT'
        # store_in_tt(zobrist_key, depth, best_value, tt_flag, best_move_for_tt)

        return best_value

# --- Hàm kiểm tra trạng thái kết thúc (Giữ nguyên hoặc cải thiện) ---
def check_terminal_state(game_state: GameState):
    # ... (Logic kiểm tra checkmate/stalemate như phiên bản trước) ...
    # Đảm bảo trả về (True, score_theo_góc_nhìn_Trắng) hoặc (False, 0.0)
    if hasattr(game_state, 'checkmate') and game_state.checkmate:
        # Bên bị chiếu hết là bên *có* lượt đi
        score = -CHECKMATE_SCORE if game_state.white_to_move else CHECKMATE_SCORE
        return True, score
    if hasattr(game_state, 'stalemate') and game_state.stalemate:
        return True, STALEMATE_SCORE
    # TODO: Thêm kiểm tra hòa khác
    return False, 0.0
# Trong Chess_AI.py

def minimax_root(game_state, ai_level, return_queue, is_maximising_player=True):
    """
    Tìm nước đi tốt nhất cho AI sử dụng minimax và đặt kết quả vào queue.
    """
    depth = get_depth_based_on_level(ai_level)
    # Kiểm tra nếu ai_level không hợp lệ
    if depth is None:
        print(f"Lỗi: Mức độ AI không hợp lệ '{ai_level}'.")
        return_queue.put(None)
        return

    # --- Chọn tập nước đi ban đầu (ở root) ---
    # Có thể dùng logic dựa trên độ sâu ban đầu ở đây
    try:
        print(f"AI (level {ai_level}, depth {depth}) is thinking...") # Thêm log
        if depth > 3:
             print("Using find_best_moves with proportion 0.75 for initial moves.")
             legal_moves = find_best_moves(game_state, model, 1) # Sử dụng model ML để giới hạn nước đi ban đầu
        else:
             print("Using find_best_moves with full proportion for initial moves.")
             legal_moves = find_best_moves(game_state, model) # Hoặc lấy tất cả: game_state.getValidMoves()

        if not legal_moves:
             print("Cảnh báo: Không tìm thấy nước đi hợp lệ nào ban đầu.")
             return_queue.put(None)
             return

        best_move_score = 9999
        best_move_found = None

        for move in legal_moves:
            if not isinstance(move, Move):
                print(f"Lỗi: Phần tử không phải là Move trong root: {move}. Bỏ qua.")
                continue

            try:
                game_state.makeMove(move)
                # Gọi minimax với độ sâu còn lại (depth - 1)
                value = minimax(depth - 1, game_state, -10000, 10000, True)
                game_state.undoMove()
            except Exception as e:
                print(f"Lỗi khi thực hiện/đánh giá nước đi {move} từ root: {e}")
                try: game_state.undoMove()
                except Exception as undo_e: print(f"Lỗi hoàn tác sau lỗi root: {undo_e}")
                value = 9999

            print(f"Move: {move}, Score: {value}") # Log điểm số từng nước đi ở root
            if value <= best_move_score:
                best_move_score = value
                best_move_found = move

        if best_move_found is None and legal_moves:
            print("Cảnh báo: Không tìm thấy best_move_found dù có legal_moves. Chọn nước đi đầu tiên.")
            best_move_found = legal_moves[0]

    except Exception as e:
        print(f"Lỗi nghiêm trọng trong minimax_root: {e}")
        import traceback
        traceback.print_exc()
        best_move_found = None

    print(f"AI found move (in minimax_root): {best_move_found} with score: {best_move_score if 'best_move_score' in locals() else 'N/A'}")
    return_queue.put(best_move_found)

def findRandomMove(valid_moves):
     """Chọn một nước đi ngẫu nhiên từ danh sách các nước đi hợp lệ."""
     if valid_moves:
         return random.choice(valid_moves)
     return None # Trả về None nếu không có nước đi hợp lệ























# def draw_board(current_board):
#     """Draw board as ASCII art
#
#     Keyword arguments:
#     current_board -- chess.Board()
#     """
#     board_str = current_board.__str__()
#     print(board_str)


# def can_checkmate(move, current_board):
#     """Return True if a move can checkmate
#
#     Keyword arguments:
#     move -- chess.Move
#     current_board -- chess.Board()
#     """
#     fen = current_board.fen()
#     future_board = chess.Board(fen)
#     future_board.push(move)
#     return future_board.is_checkmate()
#
#
# def ai_play_turn(current_board):
#     """Handdle the A.I's turn
#
#     Keyword arguments:
#     current_board -- chess.Board()
#     """
#     clear_output()
#     draw_board(current_board)
#     print('\n')
#     print("Bot is thinking...")
#     for move in current_board.legal_moves:
#         if (can_checkmate(move, current_board)):
#             current_board.push(move)
#             return
#
#     nb_moves = len(list(current_board.legal_moves))
#
#     if (nb_moves > 30):
#         current_board.push(minimax_root(4, current_board))
#     elif (nb_moves > 10 and nb_moves <= 30):
#         current_board.push(minimax_root(5, current_board))
#     else:
#         current_board.push(minimax_root(7, current_board))
#     return
#
#
# def human_play_turn(current_board):
#     """Handle the human's turn
#
#     Keyword arguments:
#     current_board = chess.Board()
#     """
#     clear_output()
#     draw_board(current_board)
#     print('\n')
#     print('\n')
#     print('number moves: ' + str(len(current_board.move_stack)))
#     move_uci = input('Enter your move: ')
#
#     try:
#         move = chess.Move.from_uci(move_uci)
#     except:
#         return human_play_turn(current_board)
#     if (move not in current_board.legal_moves):
#         return human_play_turn(current_board)
#     current_board.push(move)
#     return
#
#
# def play_game(turn, current_board):
#     """Play through the whole game
#
#     Keyword arguments:
#     turn -- True for A.I plays first
#     current_board -- chess.Board()
#     """
#     if (current_board.is_stalemate()):
#         clear_output()
#         print('Stalemate: No one wins')
#         return
#     else:
#         if (not turn):
#             if (not current_board.is_checkmate()):
#                 human_play_turn(current_board)
#                 return play_game(not turn, current_board)
#             else:
#                 clear_output()
#                 draw_board(current_board)
#                 print('A.I wins')
#                 return
#         else:
#             if (not current_board.is_checkmate()):
#                 ai_play_turn(current_board)
#                 return play_game(not turn, current_board)
#             else:
#                 clear_output()
#                 draw_board(current_board)
#                 print('Human wins')
#                 return
#
#
# def play():
#     """Init and start the game
#     """
#     global ai_white
#     ai_white = True
#
#     board = chess.Board()
#     human_first = input('Choose a colour [w/b]: ')
#     clear_output()
#     if (human_first == 'w'):
#         ai_white = False
#         return play_game(False, board)
#     else:
#         return play_game(True, board)
#
#
# play()