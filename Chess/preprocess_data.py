import chess
import chess.pgn  # Đảm bảo import đúng chess.pgn
import numpy as np
import os

def fen_to_vector(fen):
    vector = np.zeros((8, 8, 12), dtype=np.float32)
    board = chess.Board(fen)
    piece_symbols = "pnbrqkPNBRQK"

    for square, piece in board.piece_map().items():
        piece_index = piece_symbols.index(piece.symbol())
        row, col = divmod(square, 8)
        vector[row][col][piece_index] = 1

    return vector

def move_to_output(move):
    try:
        # Xử lý phong cấp (promote) nếu có
        if len(move) > 4:
            move = move[:4]  # Chỉ lấy nước đi cơ bản (bỏ ký tự phong cấp)

        start_square = chess.parse_square(move[:2])
        end_square = chess.parse_square(move[2:])
        move_index = start_square * 64 + end_square
        output = np.zeros(4096, dtype=np.int8)
        output[move_index] = 1
        return output
    except ValueError:
        print(f"Bỏ qua nước đi không hợp lệ: {move}")
        return None  # Nước đi không hợp lệ sẽ bị bỏ qua

def process_pgn_files(input_folder, output_file, batch_size=5000):
    X_data = []
    y_data = []
    batch_count = 0

    for file in os.listdir(input_folder):
        if file.endswith(".pgn"):
            print(f"Đang xử lý {file}...")
            with open(os.path.join(input_folder, file)) as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        move_str = str(move)
                        move_vector = move_to_output(move_str)
                        if move_vector is not None:
                            X_data.append(fen_to_vector(board.fen()))
                            y_data.append(move_vector)

                        # Khi đạt batch size, lưu lại dữ liệu
                        if len(X_data) >= batch_size:
                            save_batch(X_data, y_data, output_file, batch_count)
                            X_data = []
                            y_data = []
                            batch_count += 1

    # Lưu lại dữ liệu cuối cùng nếu còn
    if X_data:
        save_batch(X_data, y_data, output_file, batch_count)

    print(f"Đã lưu dữ liệu tại {output_file}_batch_*.npz")

def save_batch(X_data, y_data, output_file, batch_count):
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int8)
    np.savez_compressed(f"{output_file}_batch_{batch_count}.npz", X=X_data, y=y_data)
    print(f"Lưu batch {batch_count}: {len(X_data)} mẫu.")

# Sử dụng
process_pgn_files(r"D:\KieuQuy\Documents\AI\Chess\Chess\dataset_pgns", "chess_training_data", batch_size=5000)
