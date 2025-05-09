import numpy as np
import tensorflow as tf
import random
import json
import chess
import time
from ChessEngine import *


transposition_table = {}
tt_hits = 0
tt_probes = 0

TT_EXACT = 0
TT_LOWERBOUND = 1 # Alpha, fail-high
TT_UPPERBOUND = 2 # Beta, fail-low


CHECKMATE_SCORE = 30000
STALEMATE_SCORE = 0.0
MAX_PLY = 60
MAX_QS_PLY = 3
MAX_OPENING_MOVES_IN_AI = 8

path_to_nnue_model = 'D:/AI/Chess/ChessApp/Chess/model/train/my_chess_eval_model1.h5'
OPENING_BOOK_PATH = "D:/AI/Chess/ChessApp/Chess/pgnopening_book_from_pgn.json"

opening_book_loaded = {}
try:
    with open(OPENING_BOOK_PATH, 'r') as f:
        opening_book_loaded = json.load(f)
    print(f"Sách khai cuộc đã tải từ: {OPENING_BOOK_PATH}")
except Exception as e:
    print(f"Lỗi khi tải sách khai cuộc: {e}")


global model_nnue
model_nnue = tf.keras.models.load_model(path_to_nnue_model)


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
    return {0: (7, 0), 1: (7, 1), 2: (7, 2), 3: (7, 3), 4: (7, 4), 5: (7, 5), 6: (7, 6), 7: (7, 7),
            8: (6, 0), 9: (6, 1), 10: (6, 2), 11: (6, 3), 12: (6, 4), 13: (6, 5), 14: (6, 6), 15: (6, 7),
            16: (5, 0), 17: (5, 1), 18: (5, 2), 19: (5, 3), 20: (5, 4), 21: (5, 5), 22: (5, 6), 23: (5, 7),
            24: (4, 0), 25: (4, 1), 26: (4, 2), 27: (4, 3), 28: (4, 4), 29: (4, 5), 30: (4, 6), 31: (4, 7),
            32: (3, 0), 33: (3, 1), 34: (3, 2), 35: (3, 3), 36: (3, 4), 37: (3, 5), 38: (3, 6), 39: (3, 7),
            40: (2, 0), 41: (2, 1), 42: (2, 2), 43: (2, 3), 44: (2, 4), 45: (2, 5), 46: (2, 6), 47: (2, 7),
            48: (1, 0), 49: (1, 1), 50: (1, 2), 51: (1, 3), 52: (1, 4), 53: (1, 5), 54: (1, 6), 55: (1, 7),
            56: (0, 0), 57: (0, 1), 58: (0, 2), 59: (0, 3), 60: (0, 4), 61: (0, 5), 62: (0, 6), 63: (0, 7)}[square]


def get_piece_value_pst(piece, square):
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
    if move.piece_captured != '--':
        attacker_type = move.piece_moved[1].upper()  # 'P'
        victim_type = move.piece_captured[1].upper()  # 'N'
        score += piece_simple_value.get(victim_type, 0) * 10
        score -= piece_simple_value.get(attacker_type, 0)
    return score

def is_check_move(gs_original: GameState, move: Move):
    gs_copy = gs_original.copy()
    gs_copy.makeMove(move)
    is_check = gs_copy.king_in_check()
    return is_check

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

pieces_nnue = list('rnbqkpRNBQKP.')
piece_to_index_nnue = {p: i for i, p in enumerate(pieces_nnue)}
pgn_map_nnue = {
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k',
    '--': '.'
}

def one_hot_encode_piece_nnue(piece_symbol):
    """Mã hóa one-hot cho một ký tự quân cờ ."""
    arr = np.zeros(len(pieces_nnue), dtype=np.float32) # Dùng float32 cho TF
    index = piece_to_index_nnue.get(piece_symbol, piece_to_index_nnue['.']) # Lấy index, mặc định là '.'
    arr[index] = 1.0
    return arr

def encode_board_nnue(board_2d):
    """
    Mã hóa bàn cờ từ list 2D của GameState thành mảng (64, 13) .
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
    return encoded_array_64_13


# --- HÀM ĐÁNH GIÁ MỚI SỬ DỤNG NNUE ---
def evaluate_board_nnue(game_state: GameState, loaded_model):
    """
    Hàm đánh giá thế cờ tĩnh sử dụng mô hình đã huấn luyện.

    Args:
        game_state: Trạng thái hiện tại của trò chơi (từ ChessEngine).
        loaded_model: Mô hình Keras  đã được tải.

    Returns:
        Điểm đánh giá (float), ví dụ: centipawns.
    """
    try:
        nnue_input_flat = encode_board_nnue(game_state.board)
        nn_input = np.expand_dims(nnue_input_flat, axis=0)
        prediction = loaded_model.predict(nn_input, verbose=0)
        score = prediction[0][0]
        return float(score)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Fallback về hàm đánh giá PST...")
        return evaluate_board_pst(game_state)

def quiescence_search(game_state: GameState, alpha, beta, ply=0):
    global model_nnue, transposition_table
    if ply >= MAX_QS_PLY:
        return evaluate_board_nnue(game_state, model_nnue)
    try:
        stand_pat_score = evaluate_board_nnue(game_state, model_nnue)
    except Exception as e:
        print(f"Lỗi khi gọi evaluate_board_nnue trong QS (ply {ply}): {e}")
        return 0.0

    if game_state.white_to_move:
        if stand_pat_score >= beta:
            return beta
        alpha = max(alpha, stand_pat_score)
        best_value_found = stand_pat_score
    else:
        if stand_pat_score <= alpha:
            return alpha
        beta = min(beta, stand_pat_score)
        best_value_found = stand_pat_score

    try:
        valid_moves = game_state.getValidMoves()
        capture_moves = [move for move in valid_moves if move.piece_captured != '--']
    except Exception as e:
        print(f"Lỗi khi getValidMoves trong QS (ply {ply}): {e}")
        return stand_pat_score
    for move in capture_moves:
        try:
            game_state.makeMove(move)
            score = quiescence_search(game_state, alpha, beta, ply + 1)
            game_state.undoMove()

            if game_state.white_to_move:
                best_value_found = max(best_value_found, score)
                alpha = max(alpha, best_value_found)
            else:
                best_value_found = min(best_value_found, score)
                beta = min(beta, best_value_found)

            if alpha >= beta:
                break

        except Exception as e:
            print(f"Lỗi trong vòng lặp QS (ply {ply}) cho nước đi {move}: {e}")
            try:
                if game_state.move_log and game_state.move_log[-1] == move:
                     game_state.undoMove()
            except Exception as ue:
                print(f"Lỗi khi undoMove sau lỗi trong QS: {ue}")
            continue
    return best_value_found

def check_terminal_state(game_state: GameState, ply_from_root=0):
    CHECKMATE_SCORE = 99999
    STALEMATE_SCORE = 0.0
    """Kiểm tra trạng thái kết thúc và trả về điểm."""
    if hasattr(game_state, 'checkmate') and game_state.checkmate:

        if not game_state.white_to_move: score = CHECKMATE_SCORE - ply_from_root
        else: score = -CHECKMATE_SCORE + ply_from_root
        return True, score
    if hasattr(game_state, 'stalemate') and game_state.stalemate:
        return True, STALEMATE_SCORE
    return False, 0.0


def get_depth_based_on_level(ai_level):
    if ai_level == 'easy': return 2
    elif ai_level == 'hard': return 4


def minimax(depth, game_state: GameState, alpha, beta, is_maximising_player, ply_from_root=0):
    global model_nnue
    original_alpha = alpha
    original_beta = beta

    # --- TT Lookup ---
    current_hash = game_state.get_current_zobrist_hash()

    if current_hash in transposition_table:
        entry = transposition_table[current_hash]
        if entry['depth'] >= depth:
            if entry['flag'] == TT_EXACT:
                return entry['score']
            elif entry['flag'] == TT_LOWERBOUND:
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == TT_UPPERBOUND:
                beta = min(beta, entry['score'])

            if alpha >= beta:
                return entry['score']

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

        if hasattr(move, 'is_pawn_promotion') and move.is_pawn_promotion:
            promotion_moves.append(move)
        else:
            other_moves_after_promotions.append(move)

    def get_promotion_score(move_obj, gs):
        if not (hasattr(move_obj, 'is_pawn_promotion') and move_obj.is_pawn_promotion):
            return -1
        promoted_char = move_obj.promoted_to_piece[1].upper() if hasattr(move_obj,
                                                                         'promoted_to_piece') and move_obj.promoted_to_piece else 'Q'
        return piece_simple_value.get(promoted_char, 0)

    promotion_moves.sort(key=lambda m: get_promotion_score(m, game_state), reverse=True)
    ordered_moves.extend(promotion_moves)

    current_ordered_set = set(ordered_moves)  # Dùng set để kiểm tra nhanh
    remaining_quiet_moves = [m for m in other_moves_after_promotions if m not in current_ordered_set]
    ordered_moves.extend(remaining_quiet_moves)


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

                    break  # Cắt tỉa Beta
            except Exception as e:
                print(f"Error in minimax (max) loop, move {move}: {e}")
                try:

                    if game_state.move_log and game_state.move_log[-1].moveID == move.moveID:  # Giả sử Move có moveID
                        game_state.undoMove()
                except Exception as ue:
                    print(f"Undo error after max error: {ue}")
                continue
        # --- TT Store ---
        entry_data = {'score': best_value,'depth': depth}
        if best_value <= original_alpha:
            entry_data['flag'] = TT_UPPERBOUND
        elif best_value >= beta:
            entry_data['flag'] = TT_LOWERBOUND
        else:
            entry_data['flag'] = TT_EXACT
        transposition_table[current_hash] = entry_data
        return best_value
    else:
        best_value = float('inf')
        for move_idx, move in enumerate(ordered_moves):
            try:
                game_state.makeMove(move)
                value = minimax(depth - 1, game_state, alpha, beta, True, ply_from_root + 1)
                game_state.undoMove()

                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if alpha >= beta:
                    break
            except Exception as e:
                print(f"Error in minimax (min) loop, move {move}: {e}")
                try:
                    if game_state.move_log and game_state.move_log[-1].moveID == move.moveID:
                        game_state.undoMove()
                except Exception as ue:
                    print(f"Undo error after min error: {ue}")
                continue
        # --- TT Store ---
        entry_data = {'score': best_value,'depth': depth}
        if best_value <= original_alpha:
            entry_data['flag'] = TT_UPPERBOUND
        elif best_value >= beta:
            entry_data['flag'] = TT_LOWERBOUND
        else:
            entry_data['flag'] = TT_EXACT
        transposition_table[current_hash] = entry_data
        return best_value


def get_move_from_uci(gs: GameState, uci_string: str) -> Move | None:

    try:
        move_from_uci = chess.Move.from_uci(uci_string)
        legal_moves_list = gs.getValidMoves()
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

                    if move_from_uci.promotion is not None:
                        if hasattr(legal_move, 'is_pawn_promotion') and legal_move.is_pawn_promotion:
                            expected_promotion_symbol = chess.piece_symbol(move_from_uci.promotion).upper()
                            actual_promotion_symbol = ''

                            if hasattr(legal_move, 'promoted_to_piece') and legal_move.promoted_to_piece:
                                actual_promotion_symbol = legal_move.promoted_to_piece[1].upper()

                            if expected_promotion_symbol == actual_promotion_symbol:
                                return legal_move
                        else:
                            continue
                    else:

                        return legal_move
            except Exception as coord_err:
                print(f"Lỗi chuyển đổi tọa độ cho move {legal_move}: {coord_err}")
                continue

        print(f"Cảnh báo: Nước đi UCI '{uci_string}' từ sách không phải là nước đi hợp lệ trong thế cờ hiện tại.")
        return None
    except ValueError:
        print(f"Lỗi: Chuỗi UCI không hợp lệ từ sách: '{uci_string}'")
        return None
    except Exception as e:
        print(f"Lỗi không xác định trong get_move_from_uci cho '{uci_string}': {e}")
        return None

def findRandomMove(valid_moves):
     """Chọn một nước đi ngẫu nhiên từ danh sách các nước đi hợp lệ."""
     if valid_moves:
         return random.choice(valid_moves)
     return None


# --- Hàm phụ trợ sắp xếp nước đi ở gốc ---
def order_root_moves(gs: GameState, legal_moves, pv_move):
    ordered_moves = []

    if pv_move and pv_move in legal_moves:
        ordered_moves.append(pv_move)

        remaining_moves = [m for m in legal_moves if m != pv_move]
    else:
        remaining_moves = list(legal_moves)  # Nếu không có pv_move, bắt đầu với tất cả

    captures = [m for m in remaining_moves if m.piece_captured != '--']
    non_captures = [m for m in remaining_moves if m.piece_captured == '--']

    captures.sort(key=lambda m: score_move_mvv_lva(m, gs), reverse=True)


    ordered_moves.extend(captures)
    ordered_moves.extend(non_captures)

    return ordered_moves

def find_best_move_iddfs(game_state: GameState, ai_level, return_queue, is_maximising_player, time_limit_seconds=1000.0):

    global model_nnue, transposition_table

    start_time = time.time()
    transposition_table = {}  # Reset TT cho mỗi lần tìm kiếm gốc mới
    print(f"Time limit: {time_limit_seconds}s")

    max_depth = get_depth_based_on_level(ai_level)  # Xác định độ sâu tối đa dựa trên level
    if max_depth is None:
        print(f"Error: Invalid AI level '{ai_level}'.")
        return_queue.put(None)
        return

    # --- KIỂM TRA SÁCH KHAI CUỘC ---
    book_move_obj = None  # Reset biến nước đi từ sách
    print(
        f"[Book Check] opening_book_loaded: {bool(opening_book_loaded)}, len(move_log): {len(game_state.move_log)}, MAX_OPENING_MOVES: {MAX_OPENING_MOVES_IN_AI}")

    if opening_book_loaded and len(game_state.move_log) < MAX_OPENING_MOVES_IN_AI:

        try:
            current_fen_pos = game_state.get_board_fen_position()

            if current_fen_pos in opening_book_loaded:
                candidate_moves_with_counts = opening_book_loaded[current_fen_pos]  # List of ["move_uci", count]

                if candidate_moves_with_counts:
                    moves_uci_list = [item[0] for item in candidate_moves_with_counts]
                    counts = [item[1] for item in candidate_moves_with_counts]
                    if not moves_uci_list:  # Kiểm tra nếu list rỗng sau khi lọc
                        print("  Không có nước đi hợp lệ nào trong sách cho thế cờ này sau khi lọc.")
                    else:
                        total_count = sum(counts)
                        probabilities = [c / total_count for c in counts] if total_count > 0 else None

                        attempts = 0
                        max_attempts = len(moves_uci_list)

                        while attempts < max_attempts:
                            if probabilities:
                                selected_uci_move = random.choices(moves_uci_list, weights=probabilities, k=1)[0]
                            else:
                                selected_uci_move = random.choice(moves_uci_list)

                            potential_book_move = get_move_from_uci(game_state, selected_uci_move)

                            if potential_book_move:
                                book_move_obj = potential_book_move
                                print(f"  AI chọn nước đi hợp lệ từ sách: {selected_uci_move}")
                                break
                            else:
                                try:
                                    idx_to_remove = moves_uci_list.index(selected_uci_move)
                                    del moves_uci_list[idx_to_remove]
                                    del counts[idx_to_remove]
                                    if not moves_uci_list: break
                                    total_count = sum(counts)
                                    probabilities = [c / total_count for c in counts] if total_count > 0 else None
                                except ValueError:
                                    pass
                            attempts += 1

                        if not book_move_obj:
                            print("  Không tìm thấy nước đi hợp lệ nào từ sách sau khi thử.")

        except Exception as book_lookup_error:
            print(f"Lỗi khi tra cứu sách khai cuộc: {book_lookup_error}")
            print(f"Lỗi: {book_lookup_error}")
            import traceback
            traceback.print_exc()
    if book_move_obj:
        return_queue.put(book_move_obj)
        return

    overall_best_move = None
    best_score_so_far = -float('inf') if is_maximising_player else float('inf')
    pv_move_for_next_iter = None

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
            iter_start_time = time.time()
            ordered_root_moves = order_root_moves(game_state, initial_legal_moves,
                                                  pv_move_for_next_iter)

            best_move_this_depth = None
            current_best_score_this_depth = -float('inf') if is_maximising_player else float('inf')
            alpha = -float('inf')
            beta = float('inf')
            for i, move in enumerate(ordered_root_moves):
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_limit_seconds * 0.95:
                    print(f"Time limit approaching during depth {current_depth} root move {i + 1}. Stopping search.")
                    raise TimeoutError("Time limit reached")

                try:
                    game_state.makeMove(move)
                    value = minimax(current_depth - 1, game_state, alpha, beta, not is_maximising_player,
                                    ply_from_root=1)
                    game_state.undoMove()
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

                except Exception as e:
                    print(f"Error during minimax call from root for move {move} at depth {current_depth}: {e}")
                    try:
                        if game_state.move_log and game_state.move_log[-1] == move:
                            game_state.undoMove()
                    except Exception as ue:
                        print(f"Undo error after root minimax error: {ue}")
            iter_elapsed_time = time.time() - iter_start_time
            if best_move_this_depth is not None:
                overall_best_move = best_move_this_depth
                best_score_so_far = current_best_score_this_depth
                pv_move_for_next_iter = best_move_this_depth
                print(
                    f"IDDFS Depth {current_depth} completed in {iter_elapsed_time:.2f}s. Best Move: {overall_best_move} (Score: {best_score_so_far:.2f})")
            else:
                print(
                    f"Warning: No best move found at depth {current_depth} (possibly due to errors in branches). Keeping previous best move.")

            total_elapsed_time = time.time() - start_time
            print(f"Total time after depth {current_depth}: {total_elapsed_time:.2f}s")
            if total_elapsed_time >= time_limit_seconds:
                print(f"Time limit reached after completing depth {current_depth}.")
                break

    except TimeoutError:
        print("Search stopped due to time limit (TimeoutError captured). Using best move from last completed depth.")
    except Exception as e_iddfs:
        print(f"--- Error during IDDFS main loop ---")
        print(f"Error: {e_iddfs}")
        import traceback
        traceback.print_exc()

    final_elapsed = time.time() - start_time


    if overall_best_move is None and initial_legal_moves:
        print("Warning: No best move found (e.g., time limit before depth 1 completion). Choosing first legal move.")
        overall_best_move = initial_legal_moves[0]
    elif overall_best_move is None:
        print(
            "CRITICAL WARNING: No best move found and no legal moves available (This shouldn't happen if initial check passed)!")


    print(f"--- IDDFS Finished ---")
    if overall_best_move:
        print(f"AI finished thinking (IDDFS). Best Move: {overall_best_move} (Score: {best_score_so_far:.2f})")
    else:
        print(f"AI finished thinking (IDDFS), but failed to find a best move.")
    print(f"Total search time: {final_elapsed:.2f}s")

    return_queue.put(overall_best_move)
