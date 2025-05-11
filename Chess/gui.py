import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import chess
import threading
import subprocess
import os
import platform
import sys
from PIL import Image, ImageTk

# --- Cấu hình ---
ENGINE_PATH = "D:/AI/Chess/ChessApp/Chess/main.py"
ENGINE_DIRECTORY = os.path.dirname(ENGINE_PATH)
IMAGE_PATH = "images/"

# --- Biến toàn cục ---
board = chess.Board()
selected_square = None
human_player_color = chess.WHITE
engine_process = None

# Kích thước bàn cờ và ô cờ
SQUARE_SIZE = 60  # Kích thước mỗi ô cờ (ví dụ: 60 pixels)
BOARD_DIMENSION = 8  # 8x8
COORDINATE_SPACE = 20  # Không gian cho viền tọa độ (pixels)

# Kích thước tổng của canvas bao gồm cả viền tọa độ
CANVAS_WIDTH = BOARD_DIMENSION * SQUARE_SIZE + COORDINATE_SPACE * 2  # Thêm không gian 2 bên
CANVAS_HEIGHT = BOARD_DIMENSION * SQUARE_SIZE + COORDINATE_SPACE * 2

piece_images = {}
background_image_tk = None

GAME_MODE_PVE = "pve"
GAME_MODE_PVP = "pvp"
current_game_mode = GAME_MODE_PVE

root = None
main_menu_frame = None
game_frame = None
board_canvas = None  # Sẽ được tạo với kích thước mới
status_label = None
move_log_text = None
right_panel = None
engine_started_successfully = False

pvp_button_menu = None
pve_button_menu_main = None

def load_piece_images():
    global piece_images
    piece_types = ['P', 'N', 'B', 'R', 'Q', 'K']
    colors = ['w', 'b']
    for color_char in colors:
        for piece_type in piece_types:
            filename = os.path.join(IMAGE_PATH, f"{color_char}{piece_type}.png")
            try:
                img = Image.open(filename)
                img = img.convert("RGBA")
                # Resize ảnh cho vừa ô cờ (trừ đi một chút padding nếu muốn)
                img = img.resize((SQUARE_SIZE - 10, SQUARE_SIZE - 10), Image.Resampling.LANCZOS)
                piece_images[f"{color_char}{piece_type}"] = ImageTk.PhotoImage(img)
            except FileNotFoundError:
                print(f"Error: Image file not found: {filename}")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")


def get_piece_image_key(piece):
    if not piece: return None
    return f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"



def start_engine_async(callback_on_finish):
    def _start():
        global engine_started_successfully
        engine_started_successfully = start_engine_logic()
        if root and root.winfo_exists() and callback_on_finish:
            root.after(0, callback_on_finish)

    threading.Thread(target=_start, daemon=True).start()


def start_engine_logic():
    global engine_process
    if engine_process and engine_process.poll() is None:
        send_engine_command("quit");
        try:
            engine_process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            engine_process.kill()
        engine_process = None
    command = [sys.executable if sys.executable else "python", ENGINE_PATH]
    env = os.environ.copy()
    if platform.system() == "Linux":
        env['LD_LIBRARY_PATH'] = f".:{os.path.abspath(ENGINE_DIRECTORY)}:{env.get('LD_LIBRARY_PATH', '')}"
    elif platform.system() == "Darwin":
        env['DYLD_LIBRARY_PATH'] = f".:{os.path.abspath(ENGINE_DIRECTORY)}:{env.get('DYLD_LIBRARY_PATH', '')}"
    print(f"Starting engine: {' '.join(command)} in {ENGINE_DIRECTORY}")
    try:
        engine_process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, universal_newlines=True, cwd=ENGINE_DIRECTORY, env=env
        )
        threading.Thread(target=log_engine_stderr, daemon=True).start()
        send_engine_command("uci")
        for _ in range(20):
            line = engine_process.stdout.readline().strip()
            if not line and engine_process.poll() is not None: print("Engine terminated (uci)"); return False
            print(f"Engine init: {line}")
            if "uciok" in line: break
        else:
            print("No uciok"); return False
        send_engine_command("isready")
        for _ in range(20):
            line = engine_process.stdout.readline().strip()
            if not line and engine_process.poll() is not None: print("Engine terminated (isready)"); return False
            print(f"Engine ready: {line}")
            if "readyok" in line: break
        else:
            print("No readyok"); return False
        print("Engine started and initialized.")
        return True
    except Exception as e:
        print(f"Failed to start engine: {e}"); return False


def log_engine_stderr():
    if engine_process and engine_process.stderr:
        for line in iter(engine_process.stderr.readline, ''):
            if not (root and root.winfo_exists()): break
            print(f"Engine stderr: {line.strip()}")


def send_engine_command(command):
    if engine_process and engine_process.poll() is None:
        print(f"GUI -> Engine: {command}")
        try:
            engine_process.stdin.write(command + "\n"); engine_process.stdin.flush()
        except Exception:
            handle_engine_crash()
    elif current_game_mode == GAME_MODE_PVE:
        print("Engine not running (send_engine_command).")
        if root and root.winfo_exists(): handle_engine_crash()


def handle_engine_crash():
    if not (root and root.winfo_exists()): return
    if current_game_mode == GAME_MODE_PVE:
        if not hasattr(handle_engine_crash, "showing_error"):
            handle_engine_crash.showing_error = True
            messagebox.showerror("Engine Error", "Engine connection lost. Please restart or play PvP.")
            del handle_engine_crash.showing_error


def get_engine_move():
    if current_game_mode != GAME_MODE_PVE: return None
    if not engine_process or engine_process.poll() is not None:
        print("Engine not running (get_engine_move).");
        handle_engine_crash();
        return None
    fen = board.fen();
    send_engine_command(f"position fen {fen}");
    send_engine_command("go movetime 3000")
    best_move_uci = None
    while True:
        if engine_process.poll() is not None: print(
            "Engine terminated while thinking."); handle_engine_crash(); return None
        try:
            line = engine_process.stdout.readline().strip()
        except Exception:
            print(f"Error reading engine stdout."); handle_engine_crash(); return None
        print(f"Engine -> GUI: {line}")
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) > 1 and parts[1] not in ["(none)", "0000"]: best_move_uci = parts[1]
            break
        elif not line and (
                engine_process.poll() is not None or not (engine_process.stdout and engine_process.stdout.readable())):
            print("Engine stdout closed.");
            handle_engine_crash();
            return None
    return best_move_uci


# --- Các hàm xử lý UI và Game Logic ---
def add_move_to_log(move_san):
    if move_log_text and move_log_text.winfo_exists():
        move_number_str = f"{board.fullmove_number}. " if board.turn != chess.WHITE or (
                    board.fullmove_number == 1 and len(board.move_stack) == 1) else f"{board.fullmove_number}. ... "
        move_log_text.config(state=tk.NORMAL)
        if board.turn != chess.WHITE or (board.fullmove_number == 1 and len(board.move_stack) == 1):
            move_log_text.insert(tk.END, move_number_str)
        move_log_text.insert(tk.END, move_san + " ")
        if board.turn == chess.WHITE and len(board.move_stack) > 1: move_log_text.insert(tk.END, "\n")
        move_log_text.see(tk.END);
        move_log_text.config(state=tk.DISABLED)


def clear_move_log():
    if move_log_text and move_log_text.winfo_exists():
        move_log_text.config(state=tk.NORMAL);
        move_log_text.delete(1.0, tk.END);
        move_log_text.config(state=tk.DISABLED)


def draw_board(canvas):
    if not canvas or not canvas.winfo_exists(): return
    canvas.delete("all")

    offset_x = COORDINATE_SPACE
    offset_y = COORDINATE_SPACE

    # Vẽ các ô cờ
    for r_idx in range(BOARD_DIMENSION):  # 0-7
        for f_idx in range(BOARD_DIMENSION):  # 0-7
            sq_chess_notation = chess.square(f_idx, BOARD_DIMENSION - 1 - r_idx)  # (file, rank)

            color = "#ffcc9c" if (r_idx + f_idx) % 2 == 0 else "#d88c44"

            x1 = offset_x + f_idx * SQUARE_SIZE
            y1 = offset_y + r_idx * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

            piece = board.piece_at(sq_chess_notation)
            if piece:
                img_key = get_piece_image_key(piece)
                img_to_draw = piece_images.get(img_key)
                if img_to_draw:
                    canvas.create_image(x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2, image=img_to_draw)
                else:  # Fallback
                    canvas.create_text(x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2,
                                       text=piece.unicode_symbol(), font=("Arial", SQUARE_SIZE // 2),
                                       fill="black" if piece.color == chess.WHITE else "dim gray")

    # Vẽ nhãn tọa độ
    font_coords = ("Arial", int(COORDINATE_SPACE * 0.6))  # Kích thước font cho tọa độ
    # File (a-h) - vẽ ở dưới và trên
    for f_idx in range(BOARD_DIMENSION):
        label_char = chr(ord('a') + f_idx)
        x_pos = offset_x + f_idx * SQUARE_SIZE + SQUARE_SIZE // 2
        # Dưới bàn cờ
        canvas.create_text(x_pos, offset_y + BOARD_DIMENSION * SQUARE_SIZE + COORDINATE_SPACE // 2,
                           text=label_char, font=font_coords, fill="black")
        # Trên bàn cờ
        canvas.create_text(x_pos, offset_y - COORDINATE_SPACE // 2,
                           text=label_char, font=font_coords, fill="black")

    # Rank (1-8) - vẽ ở trái và phải
    for r_idx in range(BOARD_DIMENSION):
        label_char = str(BOARD_DIMENSION - r_idx)  # Rank 8 ở trên (r_idx=0), rank 1 ở dưới (r_idx=7)
        y_pos = offset_y + r_idx * SQUARE_SIZE + SQUARE_SIZE // 2
        # Bên trái bàn cờ
        canvas.create_text(offset_x - COORDINATE_SPACE // 2, y_pos,
                           text=label_char, font=font_coords, fill="black")
        # Bên phải bàn cờ
        canvas.create_text(offset_x + BOARD_DIMENSION * SQUARE_SIZE + COORDINATE_SPACE // 2, y_pos,
                           text=label_char, font=font_coords, fill="black")

    # Đánh dấu ô đã chọn và nước đi hợp lệ (cần điều chỉnh tọa độ với offset)
    if selected_square is not None:
        r_chess, f_chess = chess.square_rank(selected_square), chess.square_file(selected_square)
        # Chuyển rank cờ (0-7 từ dưới lên) thành r_idx canvas (0-7 từ trên xuống)
        r_idx_canvas = BOARD_DIMENSION - 1 - r_chess
        f_idx_canvas = f_chess

        x1_sel = offset_x + f_idx_canvas * SQUARE_SIZE
        y1_sel = offset_y + r_idx_canvas * SQUARE_SIZE
        x2_sel = x1_sel + SQUARE_SIZE
        y2_sel = y1_sel + SQUARE_SIZE
        canvas.create_rectangle(x1_sel, y1_sel, x2_sel, y2_sel, outline="blue", width=3, tags="selection")

        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_r_chess, to_f_chess = chess.square_rank(move.to_square), chess.square_file(move.to_square)
                to_r_idx_canvas = BOARD_DIMENSION - 1 - to_r_chess
                to_f_idx_canvas = to_f_chess

                center_x = offset_x + (to_f_idx_canvas * SQUARE_SIZE) + (SQUARE_SIZE // 2)
                center_y = offset_y + (to_r_idx_canvas * SQUARE_SIZE) + (SQUARE_SIZE // 2)
                radius = SQUARE_SIZE // 8
                fill_color = "darkred" if board.is_capture(move) else "green"
                canvas.create_oval(center_x - radius, center_y - radius,
                                   center_x + radius, center_y + radius,
                                   fill=fill_color, outline=fill_color, tags="valid_move_dot")


def on_square_click(event, canvas):
    global selected_square, board

    if (current_game_mode == GAME_MODE_PVE and board.turn != human_player_color) or board.is_game_over():
        return

    # Điều chỉnh tọa độ click để bỏ qua viền
    offset_x = COORDINATE_SPACE
    offset_y = COORDINATE_SPACE

    # Kiểm tra xem click có nằm trong vùng bàn cờ 8x8 không
    if not (offset_x <= event.x < offset_x + BOARD_DIMENSION * SQUARE_SIZE and \
            offset_y <= event.y < offset_y + BOARD_DIMENSION * SQUARE_SIZE):
        selected_square = None  # Click ra ngoài, bỏ chọn
        draw_board(canvas)
        return

    file_clicked_canvas = (event.x - offset_x) // SQUARE_SIZE
    rank_idx_canvas = (event.y - offset_y) // SQUARE_SIZE  # 0-7 từ trên xuống

    # Chuyển rank_idx_canvas thành rank cờ (0-7 từ dưới lên)
    clicked_sq_chess = chess.square(file_clicked_canvas, BOARD_DIMENSION - 1 - rank_idx_canvas)

    current_player_turn_color = board.turn
    if selected_square is None:
        piece = board.piece_at(clicked_sq_chess)
        if piece and piece.color == current_player_turn_color: selected_square = clicked_sq_chess
    else:
        promotion_piece = None;
        selected_piece = board.piece_at(selected_square)
        if selected_piece and selected_piece.piece_type == chess.PAWN:
            target_rank_for_promotion = 7 if selected_piece.color == chess.WHITE else 0
            if chess.square_rank(clicked_sq_chess) == target_rank_for_promotion: promotion_piece = chess.QUEEN
        move_attempt = chess.Move(selected_square, clicked_sq_chess, promotion=promotion_piece)
        if move_attempt in board.legal_moves:
            move_san = board.san(move_attempt);
            board.push(move_attempt);
            add_move_to_log(move_san)
            selected_square = None;
            draw_board(canvas);
            canvas.update_idletasks()
            if board.is_game_over():
                check_game_status()
            elif current_game_mode == GAME_MODE_PVP:
                update_pvp_status_label()
            elif current_game_mode == GAME_MODE_PVE and board.turn != human_player_color:
                status_label.config(text="Engine is thinking...");
                root.after(50, lambda: make_engine_move(canvas))
        else:
            piece_on_clicked_sq = board.piece_at(clicked_sq_chess)
            if piece_on_clicked_sq and piece_on_clicked_sq.color == current_player_turn_color:
                selected_square = clicked_sq_chess
            else:
                selected_square = None
    draw_board(canvas)


def make_engine_move(canvas):
    if current_game_mode != GAME_MODE_PVE or board.is_game_over() or board.turn == human_player_color:
        if board.is_game_over(): check_game_status(); return

    def _engine_thinks_thread():
        engine_move_uci = get_engine_move()
        if root.winfo_exists(): root.after(0, lambda: _process_engine_move_on_main_thread(engine_move_uci, canvas))

    status_label.config(text="Engine is thinking...")
    threading.Thread(target=_engine_thinks_thread, daemon=True).start()


def _process_engine_move_on_main_thread(engine_move_uci, canvas):
    if engine_move_uci:
        try:
            move = board.parse_uci(engine_move_uci)
            if move in board.legal_moves:
                move_san = board.san(move); board.push(move); add_move_to_log(move_san)
            else:
                messagebox.showerror("Engine Error", f"Engine illegal move: {engine_move_uci}")
        except ValueError:
            messagebox.showerror("Engine Error", f"Cannot parse engine move: {engine_move_uci}")
    elif not board.is_game_over() and current_game_mode == GAME_MODE_PVE:
        print("Engine returned no move.")
    draw_board(canvas);
    check_game_status()
    if current_game_mode == GAME_MODE_PVE and not board.is_game_over(): update_pve_status_label()


def update_pvp_status_label():
    if board.is_game_over() or not (status_label and status_label.winfo_exists()): return
    turn_str = "White" if board.turn == chess.WHITE else "Black"
    status_label.config(text=f"PvP Mode. {turn_str}'s turn.{' Check!' if board.is_check() else ''}")


def update_pve_status_label():
    if board.is_game_over() or not (status_label and status_label.winfo_exists()): return
    turn_str = "Your turn" if board.turn == human_player_color else "Engine's turn"
    status_label.config(text=f"PvE Mode. {turn_str}.{' Check!' if board.is_check() else ''}")


def check_game_status():
    if not (status_label and status_label.winfo_exists()): return
    msg_title = "Game Over";
    msg_text = "";
    game_is_over = True
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"; msg_text = f"Checkmate! {winner} wins."
    elif board.is_stalemate():
        msg_text = "Draw by Stalemate."
    else:
        game_is_over = False
    if game_is_over:
        status_label.config(text=msg_text)
        if root.winfo_exists(): messagebox.showinfo(msg_title, msg_text)
    else:
        if current_game_mode == GAME_MODE_PVP:
            update_pvp_status_label()
        else:
            update_pve_status_label()


def start_new_game_with_mode(mode):
    global board, selected_square, current_game_mode, human_player_color
    current_game_mode = mode
    if mode == GAME_MODE_PVE:
        human_player_color = chess.WHITE
        if not (engine_process and engine_process.poll() is None):
            if not start_engine_logic(): messagebox.showerror("Engine Error",
                                                              "Engine failed. Switching to PvP."); start_new_game_with_mode(
                GAME_MODE_PVP); return
    board.reset();
    selected_square = None;
    clear_move_log()
    if board_canvas and board_canvas.winfo_exists(): draw_board(board_canvas)
    if main_menu_frame and main_menu_frame.winfo_ismapped(): main_menu_frame.pack_forget()
    if game_frame and not game_frame.winfo_ismapped(): game_frame.pack(fill=tk.BOTH, expand=True)
    if current_game_mode == GAME_MODE_PVP:
        status_label.config(text="PvP Mode. White's turn.")
    else:
        status_label.config(text="PvE Mode. Your turn (White).")
    check_game_status()


def show_main_menu():
    global main_menu_frame, background_image_tk, pvp_button_menu, pve_button_menu_main
    if game_frame and game_frame.winfo_ismapped(): game_frame.pack_forget()
    if main_menu_frame is None:
        main_menu_frame = tk.Frame(root)
        try:
            img = Image.open(os.path.join(IMAGE_PATH, "background.jpg"))
            menu_width, menu_height = 800, 600
            root.geometry(f"{menu_width}x{menu_height}")
            img = img.resize((menu_width, menu_height), Image.Resampling.LANCZOS)
            background_image_tk = ImageTk.PhotoImage(img)
            tk.Label(main_menu_frame, image=background_image_tk).place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"No background image: {e}"); main_menu_frame.config(bg="lightgray")
        pvp_button_menu = tk.Button(main_menu_frame, text="Player vs Player", font=("Arial", 16, "bold"), bg="#D2B48C",
                                    fg="black", relief=tk.RAISED, borderwidth=3,
                                    command=lambda: start_new_game_with_mode(GAME_MODE_PVP))
        pvp_button_menu.place(relx=0.25, rely=0.4, anchor=tk.CENTER, width=220, height=60)
        pve_button_menu_main = tk.Button(main_menu_frame, text="Player vs AI", font=("Arial", 16, "bold"), bg="#D2B48C",
                                         fg="black", relief=tk.RAISED, borderwidth=3,
                                         command=lambda: start_new_game_with_mode(GAME_MODE_PVE))
        pve_button_menu_main.place(relx=0.25, rely=0.6, anchor=tk.CENTER, width=220, height=60)

    main_menu_frame.pack(fill=tk.BOTH, expand=True);
    update_menu_button_states()


def update_menu_button_states():
    if main_menu_frame and main_menu_frame.winfo_exists():
        if pvp_button_menu and pvp_button_menu.winfo_exists(): pvp_button_menu.config(state=tk.NORMAL)
        pve_state = tk.NORMAL if engine_started_successfully else tk.DISABLED
        if pve_button_menu_main and pve_button_menu_main.winfo_exists(): pve_button_menu_main.config(state=pve_state)


# --- Thiết lập UI chính
def create_game_interface_widgets():
    global game_frame, board_canvas, status_label, move_log_text, right_panel

    game_frame = tk.Frame(root)

    board_canvas = tk.Canvas(game_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="dim gray")  # Màu nền cho viền
    board_canvas.pack(side=tk.LEFT, padx=10, pady=10)
    board_canvas.bind("<Button-1>", lambda event: on_square_click(event, board_canvas))

    right_panel = tk.Frame(game_frame)
    right_panel.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    log_frame = tk.LabelFrame(right_panel, text="Move Log", font=("Arial", 10))
    log_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
    move_log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state=tk.DISABLED, height=20, width=25,
                                              font=("Arial", 10))
    move_log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    ttk.Button(right_panel, text="Back to Main Menu", command=show_main_menu).pack(pady=10, padx=5, fill=tk.X)

    status_label = tk.Label(right_panel, text="Status", relief=tk.SUNKEN, anchor="w", justify=tk.LEFT,
                            font=("Arial", 9))
    status_label.pack(pady=5, padx=5, fill=tk.X, side=tk.BOTTOM)


def cleanup_and_exit():
    global root, engine_process
    if engine_process and engine_process.poll() is None:
        print("Sending quit command to engine...");
        send_engine_command("quit")
        try:
            engine_process.wait(timeout=0.3)
        except:
            print("Engine cleanup timeout/error.");
        if engine_process and engine_process.poll() is None: engine_process.kill()
    if root and root.winfo_exists(): root.destroy()
    sys.exit()


# --- Khởi động ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chess")
    load_piece_images()
    create_game_interface_widgets()  # Tạo widget game trước
    show_main_menu()  # Hiển thị menu
    start_engine_async(callback_on_finish=update_menu_button_states)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("GUI interrupted."); cleanup_and_exit()
    cleanup_and_exit()