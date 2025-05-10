import os

def split_pgn_to_small_files(pgn_file, output_folder, max_size_mb=1000):
    """
    Chia file PGN lớn thành các file nhỏ (mỗi file ~1GB).
    
    Args:
        pgn_file (str): Đường dẫn đến file PGN lớn.
        output_folder (str): Thư mục để lưu các file PGN nhỏ.
        max_size_mb (int): Kích thước tối đa của mỗi file nhỏ (MB).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    file_count = 0
    current_size = 0
    new_file = open(f"{output_folder}/part_{file_count}.pgn", "w", encoding="utf-8")

    with open(pgn_file, "r", encoding="utf-8", errors="ignore") as pgn:
        game_data = ""
        valid_games = 0
        invalid_games = 0
        inside_game = False  # Để xác định ván đấu đã bắt đầu

        for line in pgn:
            # Phát hiện bắt đầu một ván đấu (các dòng metadata như [Event], [Date])
            if line.startswith("[") and "]" in line:
                if inside_game and game_data.strip():
                    # Kết thúc ván đấu trước
                    if is_valid_pgn(game_data):
                        new_file.write(game_data + "\n\n")
                        valid_games += 1
                        current_size += len(game_data.encode('utf-8')) / (1024 * 1024)  # MB
                    else:
                        invalid_games += 1
                    
                    # Chia file nếu kích thước vượt quá giới hạn
                    if current_size > max_size_mb:
                        new_file.close()
                        file_count += 1
                        current_size = 0
                        new_file = open(f"{output_folder}/part_{file_count}.pgn", "w", encoding="utf-8")
                    
                    game_data = ""

                # Bắt đầu ván đấu mới
                game_data = line
                inside_game = True
            else:
                # Thêm dòng của ván đấu vào game_data
                if inside_game:
                    game_data += line
        
        # Ghi lại ván đấu cuối cùng nếu hợp lệ
        if is_valid_pgn(game_data):
            new_file.write(game_data + "\n\n")
            valid_games += 1
        else:
            invalid_games += 1
        
        new_file.close()

    print(f"Đã chia thành {file_count + 1} file.")
    print(f"Tổng số ván đấu hợp lệ: {valid_games}")
    print(f"Tổng số ván đấu lỗi: {invalid_games}")

def is_valid_pgn(game_data):
    """
    Kiểm tra xem chuỗi PGN có phải là một ván đấu hợp lệ không.
    """
    lines = game_data.split("\n")
    moves_found = False
    for line in lines:
        # Kiểm tra dòng chứa nước đi (số lượt)
        if line.strip() and line[0].isdigit():
            moves_found = True
            break
    return moves_found and "1. " in game_data  # Phải chứa nước đi đầu tiên "1. "

# Đường dẫn tuyệt đối đến file PGN của bạn
pgn_file = r"D:\KieuQuy\Documents\AI\Chess\Chess\lichess_db_standard_rated_2025-04.pgn"
output_folder = r"D:\KieuQuy\Documents\AI\Chess\Chess\dataset_pgns"
split_pgn_to_small_files(pgn_file, output_folder, max_size_mb=1024)
