"""
Handling the AI moves.
"""
import random
import chess.engine
from pyexpat import model

import stockfish as StockFish
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import threading

piece_score = {"K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1}

knight_scores = [[0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0],
                 [0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1],
                 [0.2, 0.5, 0.6, 0.65, 0.65, 0.6, 0.5, 0.2],
                 [0.2, 0.55, 0.65, 0.7, 0.7, 0.65, 0.55, 0.2],
                 [0.2, 0.5, 0.65, 0.7, 0.7, 0.65, 0.5, 0.2],
                 [0.2, 0.55, 0.6, 0.65, 0.65, 0.6, 0.55, 0.2],
                 [0.1, 0.3, 0.5, 0.55, 0.55, 0.5, 0.3, 0.1],
                 [0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0]]

bishop_scores = [[0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
                 [0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2],
                 [0.2, 0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.2],
                 [0.2, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.2],
                 [0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.4, 0.2],
                 [0.2, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2],
                 [0.2, 0.5, 0.4, 0.4, 0.4, 0.4, 0.5, 0.2],
                 [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0]]

rook_scores = [[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
               [0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.5],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0],
               [0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.25]]

queen_scores = [[0.0, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.0],
                [0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2],
                [0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.2],
                [0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3],
                [0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3],
                [0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.2],
                [0.2, 0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.2],
                [0.0, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.0]]

pawn_scores = [[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
               [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
               [0.3, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.3],
               [0.25, 0.25, 0.3, 0.45, 0.45, 0.3, 0.25, 0.25],
               [0.2, 0.2, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2],
               [0.25, 0.15, 0.1, 0.2, 0.2, 0.1, 0.15, 0.25],
               [0.25, 0.3, 0.3, 0.0, 0.0, 0.3, 0.3, 0.25],
               [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]

piece_position_scores = {"wN": knight_scores,
                         "bN": knight_scores[::-1],
                         "wB": bishop_scores,
                         "bB": bishop_scores[::-1],
                         "wQ": queen_scores,
                         "bQ": queen_scores[::-1],
                         "wR": rook_scores,
                         "bR": rook_scores[::-1],
                         "wp": pawn_scores,
                         "bp": pawn_scores[::-1]}

CHECKMATE = 1000
STALEMATE = 0
DEPTH = 3

game_data = pd.read_excel('D:\\AI\\Chess\\ChessApp\\Chess\\games.xlsx')
# Extract the FEN notation and move columns
moves = game_data['move'].values
X = game_data['fen'].values
y = game_data['response'].values

def get_legal_features(X):
    # Convert the FEN strings into numerical features (this is your existing FEN2ARRAY function)
    legal_features = FEN2ARRAY(X)  # Convert FEN strings to a 2D numerical matrix
    return legal_features


def findBestMoveWithAI(game_state, valid_moves, ai_level, return_queue):
    stockfish_path = "D:\\AI\\Chess\\ChessApp\\Chess\\stockfish\\stockfish-windows-x86-64-avx2.exe"


    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board()

            # Reconstruct the board state from the game's move log
            for move in game_state.move_log:
                from_square = chess.square(move.start_col, 7 - move.start_row)
                to_square = chess.square(move.end_col, 7 - move.end_row)
                try:
                    board.push(chess.Move(from_square, to_square))
                except chess.IllegalMoveError:
                    print(f"Warning: Couldn't reconstruct move {move.getChessNotation()}")
                    return_queue.put(random.choice(valid_moves) if valid_moves else None)
                    return

            print(f"Current FEN: {board.fen()}")

            if not game_state.white_to_move:  # It's Black's turn (AI's turn)
                time_limit = 1.0 if ai_level == "easy" else 3.0  # Set time limit based on AI level

                # Use AI_BOT instead of Stockfish for AI's move
                # legal_features = get_legal_features(X)  # Prepare the data for the model (FEN features, etc.)
                # legal_y = y  # Corresponding valid moves
                model, legal_features, legal_y = PEPE_AI_Train(X, y, board)  # Load the trained model
                bot_move = AI_BOT(model, legal_features, legal_y, board)  # Get AI move

                # Check if the move is legal
                if bot_move not in valid_moves:
                    print("AI chose an illegal move. Selecting a valid one instead.")
                    bot_move = random.choice(valid_moves)

                return_queue.put(bot_move)

    except Exception as e:
        print(f"Error in AI move generation: {e}")
        return_queue.put(random.choice(valid_moves) if valid_moves else None)

def AI_BOT(model, legal_features, legal_y, board):
    # Dự đoán nước đi từ mô hình học máy
    bot_move = PEPE_AI_v2(model, legal_features, legal_y, board)

    if bot_move == '':
        return bot_move, board

    if bot_move not in board.legal_moves:
        print(f"AI chose an illegal move: {bot_move}. Selecting a valid one instead.")
        bot_move = random.choice([move.uci() for move in board.legal_moves])

    # Nếu không phải là thăng cấp tốt, thực hiện nước đi
    if not pawn_promotion(board, bot_move):
        board.push_san(bot_move)
    else:
        # Chọn quân để thăng cấp, ví dụ luôn chọn quân Hậu
        board.push_san(bot_move + 'q')

    return bot_move

def pawn_promotion(board, move):
    print(move)
    # get piece type from location
    square = chess.parse_square(move[:2])
    piece = board.piece_at(square)
    if len(move) != 4 or type(piece) is type(None):  # If not a valid move
        print("Invalid move:", move)
        return False
    # all_squares = [chess.square_name(square) for square in range(64)]
    else:

        # get file of first and second move
        start_rank = int(move[1])
        end_rank = int(move[3])
        # print(move)
        # print(piece)
        if piece.piece_type == chess.PAWN and (
                (start_rank == 2 and end_rank == 1) or (start_rank == 7 and end_rank == 8)):
            # this is a promotion, tag it as such:
            promotion = True
        else:
            promotion = False

    return promotion

def FEN2ARRAY(fen_array):
    piece_mapping = {'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
                     'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, '.': 0}
    if isinstance(fen_array, str):
        fen_array = [fen_array]

    feature_matrices = []

    for fen in fen_array:
        board = chess.Board(fen)
        feature_matrix = np.zeros((8, 8), dtype=int)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_value = piece_mapping[piece.symbol()]
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                feature_matrix[rank][file] = piece_value

        feature_matrices.append(feature_matrix)

    features = np.concatenate(feature_matrices, axis=0)
    return features.reshape(len(fen_array), -1)


def PEPE_AI_Train(X, y, board):
    # Huấn luyện mô hình với dữ liệu FEN và nước đi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    features = FEN2ARRAY(X_train)  # Biến đổi FEN thành mảng số
    model = RandomForestClassifier(n_estimators=100)
    n_samples = 100
    subset_features = features[:n_samples]
    subset_labels = y_train[:n_samples]
    legal_moves = [move.uci() for move in board.legal_moves]
    indices_to_keep = [i for i, move in enumerate(y_train) if move in legal_moves]
    legal_y = [move for move in y_train if move in legal_moves]  # target variable (to predict)
    X_train2 = [other_element for i, other_element in enumerate(X_train) if i in indices_to_keep]
    legal_features = FEN2ARRAY(X_train2)  # all legal moves from training data
    model.fit(legal_features, legal_y)

    return model, legal_features, legal_y



def PEPE_AI_v2(model, X, y, board):
    # Kiểm tra nếu bàn cờ ở trạng thái ban đầu, bot là quân trắng
    if board.fen() == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1':
        first_move = 'e2e4'
        print(first_move)
        return first_move
    else:
        # Dự đoán nước đi từ mô hình học máy
        board_features = FEN2ARRAY(board.fen())  # Chuyển FEN thành đặc trưng cho mô hình
        y_pred = model.predict(board_features)  # Dự đoán nước đi từ mô hình
        print("Move from Data: ", y_pred)

        # Nếu nước đi dự đoán không hợp lệ, tái huấn luyện lại mô hình dựa trên các nước đi hợp lệ
        legal_moves = [move.uci() for move in board.legal_moves]
        if y_pred[0] not in legal_moves:
            print("Setting new legal moves, please wait...")
            model, legal_features, legal_y = PEPE_AI_Train(X, y, board)
            # Dự đoán lại nước đi
            board_features = FEN2ARRAY(board.fen())
            y_pred = model.predict(board_features)
            print("Move from Data: ", y_pred)

        # Sử dụng Stockfish để đánh giá nước đi nếu là blunder
        board2 = board.copy()
        evaluation_before, engine_move = StockFish(board)
        print("Evaluation before: ", evaluation_before)
        if type(evaluation_before) == type(None):
            evaluation_before = 0
            print('Evaluation before set to 0')

        board2.push_san(y_pred[0])
        evaluation_after, engine_move2 = StockFish(board2)  # Đánh giá từ góc nhìn của đối thủ

        if type(evaluation_after) == type(None):
            evaluation_after = 0
            print('Evaluation after set to 0')

        evaluation_after = -1 * evaluation_after  # Đánh giá từ góc nhìn của bot
        dif_eval = abs(evaluation_before - evaluation_after)
        rand_nr = random.uniform(0, 1)  # Ngay cả khi là blunder, có xác suất để không phát hiện được

        if dif_eval > 200 and evaluation_before > evaluation_after:  # Nếu là blunder
            if rand_nr > 0.1:
                bot_move = engine_move  # Điều chỉnh với nước đi của Stockfish
                print("Best move, ", bot_move)
            else:
                bot_move = y_pred[0]  # Chơi blunder nếu không phát hiện
                print("Move from Data: ", bot_move)
                print("Blunder played anyway. Prob(not seeing) = ", rand_nr)
        else:
            bot_move = y_pred[0]

    return bot_move













# def findBestMoveWithStockfish(game_state, valid_moves, ai_level, return_queue):
#     stockfish_path = "D:\\AI\\Chess\\ChessApp\\Chess\\stockfish\\stockfish-windows-x86-64-avx2.exe"
#
#     try:
#         with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
#             board = chess.Board()
#
#             # Reconstruct the board state from the game's move log
#             for move in game_state.move_log:
#                 from_square = chess.square(move.start_col, 7 - move.start_row)
#                 to_square = chess.square(move.end_col, 7 - move.end_row)
#                 try:
#                     board.push(chess.Move(from_square, to_square))
#                 except chess.IllegalMoveError:
#                     print(f"Warning: Couldn't reconstruct move {move.getChessNotation()}")
#                     return_queue.put(random.choice(valid_moves) if valid_moves else None)
#                     return
#
#             print(f"Current FEN: {board.fen()}")
#
#             if not game_state.white_to_move:  # It's Black's turn (AI's turn)
#                 time_limit = 1.0 if ai_level == "easy" else 3.0  # Đặt thời gian tùy theo cấp độ AI
#                 try:
#                     result = engine.play(board, chess.engine.Limit(time=time_limit))
#                     if result.move:
#                         # Convert Stockfish move to our move format
#                         start_row = 7 - chess.square_rank(result.move.from_square)
#                         start_col = chess.square_file(result.move.from_square)
#                         end_row = 7 - chess.square_rank(result.move.to_square)
#                         end_col = chess.square_file(result.move.to_square)
#                         # Find the corresponding move in valid_moves
#                         for move in valid_moves:
#                             if (move.start_row == start_row and move.start_col == start_col and
#                                 move.end_row == end_row and move.end_col == end_col):
#                                 return_queue.put(move)
#                                 return
#                         # If no matching move found, return a random valid move
#                         return_queue.put(random.choice(valid_moves) if valid_moves else None)
#                 except chess.engine.EngineError as e:
#                     print(f"Stockfish error: {e}")
#                     return_queue.put(random.choice(valid_moves) if valid_moves else None)
#     except Exception as e:
#         print(f"Error in Stockfish AI: {e}")
#         return_queue.put(random.choice(valid_moves) if valid_moves else None)
#
#
# # def findBestMove(game_state, valid_moves,ai_level, return_queue):
# #     global next_move
# #     next_move = None
# #     depth = get_depth_for_level(ai_level)  # Xác định độ sâu tìm kiếm theo cấp độ
# #     random.shuffle(valid_moves)
# #     findMoveNegaMaxAlphaBeta(game_state, valid_moves, depth, -CHECKMATE, CHECKMATE,
# #                              1 if game_state.white_to_move else -1)
# #     return_queue.put(next_move)
#
#
# def findMoveNegaMaxAlphaBeta(game_state, valid_moves, depth, alpha, beta, turn_multiplier):
#     global next_move
#     if depth == 0:
#         return turn_multiplier * scoreBoard(game_state)
#     # move ordering - implement later //TODO
#     max_score = -CHECKMATE
#     for move in valid_moves:
#         game_state.makeMove(move)
#         next_moves = game_state.getValidMoves()
#         score = -findMoveNegaMaxAlphaBeta(game_state, next_moves, depth - 1, -beta, -alpha, -turn_multiplier)
#         if score > max_score:
#             max_score = score
#             if depth == DEPTH:
#                 next_move = move
#         game_state.undoMove()
#         if max_score > alpha:
#             alpha = max_score
#         if alpha >= beta:
#             break
#     return max_score
#
#
# def scoreBoard(game_state):
#     """
#     Score the board. A positive score is good for white, a negative score is good for black.
#     """
#     if game_state.checkmate:
#         if game_state.white_to_move:
#             return -CHECKMATE  # black wins
#         else:
#             return CHECKMATE  # white wins
#     elif game_state.stalemate:
#         return STALEMATE
#     score = 0
#     for row in range(len(game_state.board)):
#         for col in range(len(game_state.board[row])):
#             piece = game_state.board[row][col]
#             if piece != "--":
#                 piece_position_score = 0
#                 if piece[1] != "K":
#                     piece_position_score = piece_position_scores[piece][row][col]
#                 if piece[0] == "w":
#                     score += piece_score[piece[1]] + piece_position_score
#                 if piece[0] == "b":
#                     score -= piece_score[piece[1]] + piece_position_score
#
#     return score
#
#
# def findRandomMove(valid_moves):
#     """
#     Picks and returns a random valid move.
#     """
#     return random.choice(valid_moves)

