import tkinter as tk
from tkinter import messagebox
import chess
import threading
import subprocess
import os
import platform
from PIL import Image, ImageTk # Thêm Pillow

# --- Cấu hình Engine ---
ENGINE_PATH = "D:/AI/Chess/ChessApp/Chess/main1.py"
ENGINE_DIRECTORY = os.path.dirname(ENGINE_PATH)
IMAGE_PATH = "images/" # Thư mục chứa ảnh quân cờ

# --- Biến toàn cục ---
board = chess.Board()
selected_square = None
human_player_color = chess.WHITE
engine_process = None
ui_board_size = 480 # Kích thước bàn cờ trên UI (ví dụ 60px/ô)
square_size = ui_board_size // 8


piece_images = {}

def load_piece_images():
    """Tải tất cả hình ảnh quân cờ và lưu vào piece_images."""
    global piece_images
    piece_types = ['P', 'N', 'B', 'R', 'Q', 'K']
    colors = ['w', 'b']
    for color in colors:
        for piece_type in piece_types:
            filename = f"{IMAGE_PATH}{color}{piece_type}.png"
            try:
                img = Image.open(filename)
                img = img.convert("RGBA")
                img = img.resize((square_size - 5, square_size - 5), Image.Resampling.LANCZOS)
                piece_images[f"{color}{piece_type}"] = ImageTk.PhotoImage(img)
            except FileNotFoundError:
                print(f"Error: Image file not found: {filename}")
                piece_images[f"{color}{piece_type}"] = None
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                piece_images[f"{color}{piece_type}"] = None

def get_piece_image_key(piece):
    """Trả về key để truy cập ảnh quân cờ từ piece_images."""
    if not piece:
        return None
    color_char = 'w' if piece.color == chess.WHITE else 'b'
    piece_char = piece.symbol().upper()
    return f"{color_char}{piece_char}"


def start_engine():
    global engine_process
    if engine_process and engine_process.poll() is None:
        engine_process.kill()

    command = [sys.executable if sys.executable else "python", ENGINE_PATH]

    env = os.environ.copy()
    if platform.system() == "Linux":
        env['LD_LIBRARY_PATH'] = f".:{os.path.abspath(ENGINE_DIRECTORY)}:{env.get('LD_LIBRARY_PATH', '')}"
    elif platform.system() == "Darwin": # macOS
        env['DYLD_LIBRARY_PATH'] = f".:{os.path.abspath(ENGINE_DIRECTORY)}:{env.get('DYLD_LIBRARY_PATH', '')}"

    print(f"Starting engine with command: {' '.join(command)}")
    print(f"Working directory for engine: {ENGINE_DIRECTORY}")

    try:
        engine_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=ENGINE_DIRECTORY,
            env=env
        )

        for _ in range(10): # Thử đọc 10 dòng hoặc timeout
            line = engine_process.stdout.readline().strip()
            if not line and engine_process.poll() is not None: break # Engine đã dừng
            print(f"Engine init: {line}")
            if "uciok" in line: break
        send_engine_command("isready")
        for _ in range(10):
            line = engine_process.stdout.readline().strip()
            if not line and engine_process.poll() is not None: break
            print(f"Engine ready: {line}")
            if "readyok" in line: break
        print("Engine started and initialized.")

    except FileNotFoundError:
        messagebox.showerror("Engine Error", f"Engine script not found at {ENGINE_PATH}\nMake sure habu.py is in the correct location.")
        root.quit()
    except Exception as e:
        messagebox.showerror("Engine Error", f"Failed to start engine: {e}")
        root.quit()


def send_engine_command(command):
    if engine_process and engine_process.poll() is None:
        print(f"GUI -> Engine: {command}")
        engine_process.stdin.write(command + "\n")
        engine_process.stdin.flush()
    else:
        print("Engine process not running or has terminated.")
        if not root.winfo_exists(): return

def get_engine_move():
    if not engine_process or engine_process.poll() is not None:
        print("Engine not running or has terminated.")
        if engine_process and engine_process.stderr:
            try:
                stderr_output = engine_process.stderr.read()
                if stderr_output:
                    print(f"Engine stderr output:\n{stderr_output}")
            except Exception as e:
                print(f"Error reading engine stderr: {e}")
        return None

    fen = board.fen()
    send_engine_command(f"position fen {fen}")
    send_engine_command("go movetime 2000")

    best_move_uci = None
    while True:
        if engine_process.poll() is not None:
            print("Engine terminated unexpectedly while thinking.")
            if engine_process.stderr:
                try:
                    for err_line in engine_process.stderr.readlines():
                        print(f"Engine stderr: {err_line.strip()}")
                except: pass
            return None

        try:
            line = engine_process.stdout.readline().strip()
        except Exception as e:
            print(f"Error reading engine stdout: {e}")
            return None

        print(f"Engine -> GUI: {line}")
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) > 1:
                best_move_uci = parts[1]
            break
        elif not line and (engine_process.poll() is not None or not engine_process.stdout.readable()):
            print("Engine stdout closed or process terminated.")
            return None
    return best_move_uci

# --- Các hàm xử lý UI ---
def draw_board(canvas):
    canvas.delete("all")

    # Vẽ các ô cờ
    for r in range(8):
        for f in range(8):
            sq_chess_notation = chess.square(f, 7 - r) # chess.square(file, rank)
            color = "white" if (r + f) % 2 == 0 else "lightgray" # Màu ô cờ
            x1, y1 = f * square_size, r * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

            # Vẽ quân cờ bằng ảnh
            piece = board.piece_at(sq_chess_notation)
            if piece:
                img_key = get_piece_image_key(piece)
                if img_key and piece_images.get(img_key):
                    # Căn giữa ảnh trong ô
                    canvas.create_image(x1 + square_size // 2,
                                       y1 + square_size // 2,
                                       image=piece_images[img_key])
                else: # Fallback nếu không có ảnh
                    piece_symbol = piece.unicode_symbol()
                    font_size = square_size // 2
                    canvas.create_text(x1 + square_size // 2,
                                       y1 + square_size // 2,
                                       text=piece_symbol,
                                       font=("Arial", font_size),
                                       fill="black" if piece.color == chess.WHITE else "dim gray")

    # Đánh dấu ô đã chọn
    if selected_square is not None:
        r, f = chess.square_rank(selected_square), chess.square_file(selected_square)
        x1, y1 = f * square_size, (7 - r) * square_size
        x2, y2 = x1 + square_size, y1 + square_size
        canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=3, tags="selection")

        # Đánh dấu các nước đi hợp lệ
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_r, to_f = chess.square_rank(move.to_square), chess.square_file(move.to_square)
                center_x = (to_f * square_size) + (square_size // 2)
                center_y = ((7 - to_r) * square_size) + (square_size // 2)
                radius = square_size // 8 # Bán kính của chấm tròn
                if board.is_capture(move):
                    canvas.create_oval(center_x - radius, center_y - radius,
                                       center_x + radius, center_y + radius,
                                       fill="darkred", outline="darkred", tags="valid_move_dot")
                else:
                    canvas.create_oval(center_x - radius, center_y - radius,
                                       center_x + radius, center_y + radius,
                                       fill="green", outline="green", tags="valid_move_dot")


def on_square_click(event, canvas):
    global selected_square, board

    if board.turn != human_player_color or board.is_game_over():
        return

    file_clicked = event.x // square_size
    rank_clicked = 7 - (event.y // square_size)
    clicked_sq = chess.square(file_clicked, rank_clicked)

    if selected_square is None:
        piece = board.piece_at(clicked_sq)
        if piece and piece.color == human_player_color:
            selected_square = clicked_sq
    else:
        promotion_piece = None
        if board.piece_at(selected_square) and \
           board.piece_at(selected_square).piece_type == chess.PAWN and \
           (chess.square_rank(clicked_sq) == 0 or chess.square_rank(clicked_sq) == 7):
            promotion_piece = chess.QUEEN # TODO: Cho phép người dùng chọn

        move = chess.Move(selected_square, clicked_sq, promotion=promotion_piece)

        if move in board.legal_moves:
            board.push(move)
            selected_square = None
            draw_board(canvas) # Vẽ lại bàn cờ sau nước đi của người
            canvas.update()
            if not board.is_game_over():
                root.after(100, lambda: make_engine_move(canvas))
        else: # Click không hợp lệ hoặc vào ô khác của mình
            piece_on_clicked_sq = board.piece_at(clicked_sq)
            if piece_on_clicked_sq and piece_on_clicked_sq.color == human_player_color:
                selected_square = clicked_sq # Chọn quân khác của mình
            else:
                selected_square = None # Bỏ chọn
    draw_board(canvas)


def make_engine_move(canvas):
    if board.is_game_over():
        check_game_status()
        return

    def _engine_thinks():
        engine_move_uci = get_engine_move()
        if engine_move_uci:
            try:
                move = board.parse_uci(engine_move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print(f"Engine proposed an illegal move: {engine_move_uci}. Board state: {board.fen()}")
            except ValueError as e:
                print(f"Error parsing engine move {engine_move_uci}: {e}")
        root.after(0, lambda: after_engine_move_update(canvas))

    status_label.config(text="Engine is thinking...")
    canvas.update()
    engine_thread = threading.Thread(target=_engine_thinks)
    engine_thread.start()

def after_engine_move_update(canvas):
    draw_board(canvas)
    check_game_status()
    if board.turn == human_player_color and not board.is_game_over():
        status_label.config(text="Your turn.")
    elif not board.is_game_over():
         status_label.config(text="Waiting for engine...")


def check_game_status():
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        messagebox.showinfo("Game Over", f"Checkmate! {winner} wins.")
        status_label.config(text=f"Checkmate! {winner} wins.")
    elif board.is_stalemate():
        messagebox.showinfo("Game Over", "Draw by Stalemate.")
        status_label.config(text="Draw by Stalemate.")
    elif board.is_insufficient_material():
        messagebox.showinfo("Game Over", "Draw by Insufficient Material.")
        status_label.config(text="Draw by Insufficient Material.")
    elif board.is_seventyfive_moves():
        messagebox.showinfo("Game Over", "Draw by 75-move rule.")
        status_label.config(text="Draw by 75-move rule.")
    elif board.is_fivefold_repetition():
        messagebox.showinfo("Game Over", "Draw by Fivefold Repetition.")
        status_label.config(text="Draw by Fivefold Repetition.")

def new_game():
    global board, selected_square
    board.reset()
    selected_square = None

    if human_player_color == chess.WHITE:
        status_label.config(text="New game. Your turn (White).")
    else:
        status_label.config(text="New game. Engine's turn (White).")

    draw_board(board_canvas)

    if human_player_color == chess.BLACK and board.turn == chess.WHITE:
        root.after(100, lambda: make_engine_move(board_canvas))


def choose_side(color_str):
    global human_player_color
    if color_str == "white":
        human_player_color = chess.WHITE
    else:
        human_player_color = chess.BLACK
    new_game()

# --- Thiết lập UI ---
root = tk.Tk()
root.title("Habu Chess UI")


load_piece_images()

board_canvas = tk.Canvas(root, width=ui_board_size, height=ui_board_size, bg="white")
board_canvas.pack(side=tk.LEFT, padx=10, pady=10)
board_canvas.bind("<Button-1>", lambda event: on_square_click(event, board_canvas))

control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

new_game_button = tk.Button(control_frame, text="New Game", command=new_game)
new_game_button.pack(pady=5, fill=tk.X)

play_white_button = tk.Button(control_frame, text="Play as White", command=lambda: choose_side("white"))
play_white_button.pack(pady=5, fill=tk.X)

play_black_button = tk.Button(control_frame, text="Play as Black", command=lambda: choose_side("black"))
play_black_button.pack(pady=5, fill=tk.X)

status_label = tk.Label(control_frame, text="Welcome!", relief=tk.SUNKEN, anchor="w")
status_label.pack(pady=10, fill=tk.X, side=tk.BOTTOM)

# --- Khởi động ---
import sys

if __name__ == "__main__":
    start_engine()
    new_game() # Gọi new_game để thiết lập trạng thái ban đầu đúng cách

    def on_closing():
        if engine_process and engine_process.poll() is None:
            send_engine_command("quit")
            try:
                engine_process.wait(timeout=1) # Giảm timeout để không treo lâu
            except subprocess.TimeoutExpired:
                print("Engine did not quit in time, killing.")
                engine_process.kill()
            except Exception as e:
                print(f"Error during engine quit: {e}")
                if engine_process.poll() is None: engine_process.kill()

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

    # Đảm bảo engine được dừng khi UI đóng (dù có lỗi)
    if engine_process and engine_process.poll() is None:
        print("Terminating engine process forcibly...")
        engine_process.kill()