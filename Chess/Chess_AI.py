from collections import OrderedDict
from operator import itemgetter
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from ChessEngine import *

path_to_model = 'D:/AI/Chess/ChessApp/Chess/model/morphy'

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

# def predict(df_eval, imported_model):
#     """Return array of predictions for each row of df_eval
#
#     Keyword arguments:
#     df_eval -- pd.DataFrame
#     imported_model -- tf.saved_model
#     """
#     col_names = df_eval.columns
#     dtypes = df_eval.dtypes
#     predictions = []
#     for _, row in df_eval.iterrows():  # Use the underscore to discard the row index
#         example = tf.train.Example()
#         for col_name, dtype in dtypes.items():  # Loop through column names and data types
#             value = row[col_name]  # Access the DataFrame column using col_name
#             if dtype == 'object':
#                 value = bytes(value, 'utf-8')
#                 example.features.feature[col_name].bytes_list.value.extend([value])
#             elif dtype == 'float':
#                 example.features.feature[col_name].float_list.value.extend([value])
#             elif dtype == 'int':
#                 example.features.feature[col_name].int64_list.value.extend([value])
#         predictions.append(imported_model.signatures['predict'](examples=tf.constant([example.SerializeToString()])))
#     return predictions
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


# def get_possible_moves_data(current_board):
#     """Trả về pd.DataFrame của tất cả các nước đi hợp lệ dùng cho dự đoán
#
#     Keyword arguments:
#     current_board -- GameState (chứ không phải chess.Board)
#     """
#     data = []
#     moves = current_board.getValidMoves()  # Lấy các nước đi hợp lệ từ GameState
#
#     # Các tên ô từ GameState, thay thế chess.SQUARE_NAMES
#     board_feature_names = [f"{chr(col + 97)}{8 - row}" for row in range(8) for col in range(8)]
#
#     move_from_feature_names = ['from_' + square for square in board_feature_names]
#     move_to_feature_names = ['to_' + square for square in board_feature_names]
#
#     for move in moves:
#         from_square, to_square = get_move_features(move)  # Lấy các đặc trưng của nước đi
#         row = np.concatenate(
#             (get_board_features(current_board), from_square, to_square))  # Ghép các đặc trưng lại với nhau
#         data.append(row)
#
#     columns = board_feature_names + move_from_feature_names + move_to_feature_names
#
#     # Tạo DataFrame từ dữ liệu
#     df = pd.DataFrame(data=data, columns=columns)
#
#     # Chuyển đổi kiểu dữ liệu của các cột nước đi
#     for column in move_from_feature_names:
#         df[column] = df[column].astype(float)
#     for column in move_to_feature_names:
#         df[column] = df[column].astype(float)
#
#     return df


# def find_best_moves(game_state, model, proportion=1):
#     """Trả về danh sách các nước đi tốt nhất (Move) cho một đối tượng GameState."""
#     moves = game_state.getValidMoves()  # Lấy các nước đi hợp lệ từ GameState
#     df_eval = get_possible_moves_data(game_state)  # Lấy dữ liệu cho các nước đi từ GameState
#     predictions = predict(df_eval, model)  # Dự đoán xác suất các nước đi
#     good_move_probas = []
#
#     for prediction in predictions:
#         proto_tensor = tf.make_tensor_proto(prediction['probabilities'])
#         proba = tf.make_ndarray(proto_tensor)[0][1]  # Lấy xác suất của nước đi tốt
#         good_move_probas.append(proba)
#
#     # Tạo từ điển với Move là khóa và xác suất là giá trị
#     dict_ = {}
#     for move, proba in zip(moves, good_move_probas):
#         dict_[move] = proba  # move là đối tượng Move, không phải chuỗi
#
#     dict_ = OrderedDict(sorted(dict_.items(), key=itemgetter(1), reverse=True))
#
#     best_moves = list(dict_.keys())  # Lấy danh sách các nước đi đã sắp xếp
#     best_moves_to_return = best_moves[0:int(np.ceil(len(best_moves) * proportion))]
#
#     if not best_moves_to_return:
#         best_moves_to_return = [random.choice(moves)]  # Nếu không có nước đi tốt, chọn ngẫu nhiên
#
#     return best_moves_to_return

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
        return 4



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


# def minimax(ai_level, game_state, alpha, beta, is_maximising_player):
#     """Minimax algorithm with alpha-beta pruning"""
#     depth = get_depth_based_on_level(ai_level)
#     if depth == 0:
#         return -evaluate_board(game_state)  # Evaluate the current board state
#     elif depth > 3:
#         legal_moves = find_best_moves(game_state, model, 0.75)  # Best moves based on model
#     else:
#         legal_moves = game_state.getValidMoves()  # Use GameState's getValidMoves method
#
#     if is_maximising_player:
#         best_move = -9999
#         for move in legal_moves:
#             game_state.makeMove(move)  # Apply the move
#             best_move = max(best_move, minimax(depth - 1, game_state, alpha, beta, not is_maximising_player))
#             game_state.undoMove()  # Undo the move
#             alpha = max(alpha, best_move)
#             if beta <= alpha:  # Beta cut-off
#                 return best_move
#         return best_move
#     else:
#         best_move = 9999
#         for move in legal_moves:
#             game_state.makeMove(move)  # Apply the move
#             best_move = min(best_move, minimax(depth - 1, game_state, alpha, beta, not is_maximising_player))
#             game_state.undoMove()  # Undo the move
#             beta = min(beta, best_move)
#             if beta <= alpha:  # Alpha cut-off
#                 return best_move
#         return best_move
#
#
#
# def minimax_root(game_state, ai_level, return_queue, is_maximising_player=True):
#     """
#     Tìm nước đi tốt nhất cho AI sử dụng thuật toán minimax.
#
#     game_state: đối tượng GameState chứa trạng thái bàn cờ
#     depth: độ sâu của tìm kiếm minimax
#     is_maximising_player: True nếu AI đang tìm kiếm nước đi tối ưu cho quân của mình
#     """
#     depth = get_depth_based_on_level(ai_level)
#     legal_moves = find_best_moves(game_state, model)  # Lấy các nước đi tốt nhất
#     best_move = -9999
#     best_move_found = None
#
#     for move in legal_moves:
#         # Đảm bảo move là đối tượng Move
#         if not isinstance(move, Move):
#             raise TypeError("move phải là đối tượng Move.")
#
#         game_state.makeMove(move)  # Thực hiện nước đi trên GameState
#         value = minimax(depth - 1, game_state, -10000, 10000, not is_maximising_player)
#         game_state.undoMove()  # Hoàn tác nước đi trên GameState
#         if value >= best_move:
#             best_move = value
#             best_move_found = move
#
#     return_queue.put(best_move_found)

# Trong Chess_AI.py

# Giả sử model và các hàm phụ trợ khác đã được định nghĩa đúng
# Giả sử Move được import từ ChessEngine

def minimax(depth, game_state, alpha, beta, is_maximising_player): # Sửa: Nhận depth
    """Minimax algorithm with alpha-beta pruning"""
    # Xóa: depth = get_depth_based_on_level(ai_level)

    # Trường hợp cơ sở: độ sâu bằng 0
    if depth == 0:
        # Giả sử evaluate_board hoạt động đúng
        return -evaluate_board(game_state)

    # --- Logic lấy nước đi ---
    # Trong các bước đệ quy, thường xem xét tất cả nước đi hợp lệ
    # Logic chọn lọc (dùng find_best_moves) thường chỉ áp dụng ở root hoặc độ sâu nông
    legal_moves = game_state.getValidMoves() # Lấy tất cả nước đi hợp lệ

    # Xử lý trường hợp không còn nước đi (có thể là checkmate/stalemate ở nút lá)
    if not legal_moves:
         return -evaluate_board(game_state) # Trả về đánh giá tĩnh

    if is_maximising_player:
        best_value = -9999 # Đổi tên để rõ ràng hơn
        for move in legal_moves:
            if not isinstance(move, Move): continue # Kiểm tra lại kiểu dữ liệu

            try:
                game_state.makeMove(move)
                # Gọi đệ quy với depth - 1
                best_value = max(best_value, minimax(depth - 1, game_state, alpha, beta, not is_maximising_player))
                game_state.undoMove()
            except Exception as e:
                 print(f"Lỗi trong minimax (max) khi xử lý {move}: {e}")
                 # Cân nhắc hoàn tác nếu lỗi xảy ra sau makeMove
                 try: game_state.undoMove()
                 except: pass
                 continue # Bỏ qua nước đi lỗi

            alpha = max(alpha, best_value)
            if beta <= alpha:  # Beta cut-off
                break # Không cần break, chỉ cần return
        return best_value # Trả về giá trị tốt nhất tìm được
    else: # Minimising player
        best_value = 9999
        for move in legal_moves:
            if not isinstance(move, Move): continue

            try:
                game_state.makeMove(move)
                 # Gọi đệ quy với depth - 1
                best_value = min(best_value, minimax(depth - 1, game_state, alpha, beta, not is_maximising_player))
                game_state.undoMove()
            except Exception as e:
                 print(f"Lỗi trong minimax (min) khi xử lý {move}: {e}")
                 try: game_state.undoMove()
                 except: pass
                 continue

            beta = min(beta, best_value)
            if beta <= alpha:  # Alpha cut-off
                break
        return best_value

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
             legal_moves = find_best_moves(game_state, model, 0.75) # Sử dụng model ML để giới hạn nước đi ban đầu
        else:
             print("Using find_best_moves with full proportion for initial moves.")
             legal_moves = find_best_moves(game_state, model) # Hoặc lấy tất cả: game_state.getValidMoves()

        if not legal_moves:
             print("Cảnh báo: Không tìm thấy nước đi hợp lệ nào ban đầu.")
             return_queue.put(None)
             return

        best_move_score = -9999
        best_move_found = None

        for move in legal_moves:
            if not isinstance(move, Move):
                print(f"Lỗi: Phần tử không phải là Move trong root: {move}. Bỏ qua.")
                continue

            try:
                game_state.makeMove(move)
                # Gọi minimax với độ sâu còn lại (depth - 1)
                value = minimax(depth - 1, game_state, -10000, 10000, not is_maximising_player)
                game_state.undoMove()
            except Exception as e:
                print(f"Lỗi khi thực hiện/đánh giá nước đi {move} từ root: {e}")
                try: game_state.undoMove()
                except Exception as undo_e: print(f"Lỗi hoàn tác sau lỗi root: {undo_e}")
                value = -9999

            print(f"Move: {move}, Score: {value}") # Log điểm số từng nước đi ở root
            if value >= best_move_score:
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