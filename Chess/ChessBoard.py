"""
Main driver file.
Handling user input.
Displaying current GameStatus object.
"""
import multiprocessing

import pygame as p
import ChessEngine, Chess_AI2
import sys
from multiprocessing import Process, Queue, freeze_support


BOARD_WIDTH = BOARD_HEIGHT = 512
MOVE_LOG_PANEL_WIDTH = 250
MOVE_LOG_PANEL_HEIGHT = 380
DIMENSION = 8
SQUARE_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

# Kích thước cửa sổ
WIDTH, HEIGHT = BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT
BUTTON_WIDTH, BUTTON_HEIGHT = 200, 50
scroll_y = 0
# Màu sắc
WHITE = p.Color("white")
BLACK = p.Color("black")
GREY = p.Color("grey")



buttons_main = {
    "Start Game": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 4 - BUTTON_HEIGHT // 2 + 20),
    "Quit": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 2 + 30 - BUTTON_HEIGHT // 2)
}


buttons_game_mode = {
    "Player vs Player": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 4 - BUTTON_HEIGHT // 2 + 20),
    "Player vs AI": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 2 + 30 - BUTTON_HEIGHT // 2),
}

buttons_ai_level = {
    "Easy": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 4 - BUTTON_HEIGHT // 2 + 20),
    "Hard": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 2 + 30 - BUTTON_HEIGHT // 2),
}

def draw_button(screen, text, x, y, font, wood_tex): # Thêm font, wood_tex
    screen.blit(wood_tex, (x, y)) # Sử dụng wood_tex được truyền vào
    light = (200, 180, 150)
    dark  = ( 80,  60,  30)
    p.draw.line(screen, light, (x, y), (x+BUTTON_WIDTH, y), 3)
    p.draw.line(screen, light, (x, y), (x, y+BUTTON_HEIGHT), 3)
    p.draw.line(screen, dark, (x, y+BUTTON_HEIGHT), (x+BUTTON_WIDTH, y+BUTTON_HEIGHT), 3)
    p.draw.line(screen, dark, (x+BUTTON_WIDTH, y), (x+BUTTON_WIDTH, y+BUTTON_HEIGHT), 3)

    text_surf = font.render(text, True, (250, 250, 247)) # Sử dụng font được truyền vào
    shadow = font.render(text, True, (0, 0, 0)) # Sử dụng font được truyền vào
    tx, ty = x + BUTTON_WIDTH//2, y + BUTTON_HEIGHT//2

    shadow_rect = shadow.get_rect(center=(tx+2, ty+2))
    screen.blit(shadow, shadow_rect)
    text_rect = text_surf.get_rect(center=(tx, ty))
    screen.blit(text_surf, text_rect)

# Sửa các hàm menu để nhận screen, font, background, wood_tex
def main_menu(screen, font, background, wood_tex, buttons_main): # Thêm các tham số
    while True:
        screen.fill(WHITE)
        screen.blit(background, (0, 0)) # Sử dụng background được truyền vào

        for button_text, (x, y) in buttons_main.items():
            # Truyền các tham số cần thiết cho draw_button
            draw_button(screen, button_text, x, y, font, wood_tex)

        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                # Kiểm tra va chạm nút (giữ nguyên logic)
                if buttons_main["Start Game"][0] <= mouse_x <= buttons_main["Start Game"][0] + BUTTON_WIDTH and \
                        buttons_main["Start Game"][1] <= mouse_y <= buttons_main["Start Game"][1] + BUTTON_HEIGHT:
                    print("Starting game...")
                    return "start_game"
                elif buttons_main["Quit"][0] <= mouse_x <= buttons_main["Quit"][0] + BUTTON_WIDTH and \
                        buttons_main["Quit"][1] <= mouse_y <= buttons_main["Quit"][1] + BUTTON_HEIGHT:
                    print("Exiting game...")
                    p.quit()
                    sys.exit()

            p.display.flip()


def game_mode_menu(screen, font, background, wood_tex, buttons_game_mode): # Thêm các tham số
    while True:
        screen.fill(WHITE)
        screen.blit(background, (0, 0)) # Sử dụng background

        for button_text, (x, y) in buttons_game_mode.items():
            draw_button(screen, button_text, x, y, font, wood_tex) # Truyền tham số

        for event in p.event.get():
            # ... (Xử lý sự kiện và trả về như cũ) ...
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                 mouse_x, mouse_y = event.pos
                 if buttons_game_mode["Player vs Player"][0] <= mouse_x <= buttons_game_mode["Player vs Player"][0] + BUTTON_WIDTH and buttons_game_mode["Player vs Player"][1] <= mouse_y <= buttons_game_mode["Player vs Player"][1] + BUTTON_HEIGHT:
                     print("Player vs Player mode")
                     return "player_vs_player"
                 elif buttons_game_mode["Player vs AI"][0] <= mouse_x <= buttons_game_mode["Player vs AI"][0] + BUTTON_WIDTH and buttons_game_mode["Player vs AI"][1] <= mouse_y <= buttons_game_mode["Player vs AI"][1] + BUTTON_HEIGHT:
                     print("Player vs AI mode")
                     return "player_vs_ai"

            p.display.flip()


def ai_level_menu(screen, font, background, wood_tex, buttons_ai_level): # Thêm các tham số
    while True:
        screen.fill(WHITE)
        screen.blit(background, (0, 0)) # Sử dụng background

        for button_text, (x, y) in buttons_ai_level.items():
            draw_button(screen, button_text, x, y, font, wood_tex) # Truyền tham số

        for event in p.event.get():
            # ... (Xử lý sự kiện và trả về như cũ) ...
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                 mouse_x, mouse_y = event.pos
                 if buttons_ai_level["Easy"][0] <= mouse_x <= buttons_ai_level["Easy"][0] + BUTTON_WIDTH and buttons_ai_level["Easy"][1] <= mouse_y <= buttons_ai_level["Easy"][1] + BUTTON_HEIGHT:
                     print("AI level: Easy")
                     return "easy"
                 elif buttons_ai_level["Hard"][0] <= mouse_x <= buttons_ai_level["Hard"][0] + BUTTON_WIDTH and buttons_ai_level["Hard"][1] <= mouse_y <= buttons_ai_level["Hard"][1] + BUTTON_HEIGHT:
                     print("AI level: Hard")
                     return "hard"

            p.display.flip()

def loadImages():
    """
    Initialize a global directory of images.
    This will be called exactly once AFTER pygame init.
    """
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        try:
            # Chỉ load, chưa scale hay convert ở đây nếu muốn linh hoạt
            # Hoặc nếu chắc chắn dùng scale thì làm luôn:
            image = p.image.load("images/" + piece + ".png")
            IMAGES[piece] = p.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
        except p.error as e:
            print(f"Lỗi tải ảnh {piece}: {e}")
            # Có thể thêm ảnh mặc định hoặc xử lý lỗi khác
            IMAGES[piece] = p.Surface((SQUARE_SIZE, SQUARE_SIZE)) # Tạo surface trống nếu lỗi
            IMAGES[piece].fill(GREY)

def run_game(screen, clock, font, move_log_font, player_one, player_two, ai_level=None):
    """Chạy một phiên chơi cờ."""
    game_state = ChessEngine.GameState()
    valid_moves = game_state.getValidMoves()
    move_made = False
    animate = False
    # loadImages() # Tải ảnh một lần TRƯỚC KHI gọi run_game
    global scroll_y  # Khai báo để drawMoveLog có thể dùng scroll_y
    scroll_y = 0
    square_selected = ()
    player_clicks = []
    game_over = False
    ai_thinking = False
    move_undone = False
    move_finder_process = None
    end_game_text = "" # Khởi tạo end_game_text

    running = True
    while running:
        human_turn = (game_state.white_to_move and player_one) or (not game_state.white_to_move and player_two)

        # --- Xử lý sự kiện ---
        for e in p.event.get():
            handle_scroll(e) # Đảm bảo handle_scroll được định nghĩa hoặc xóa nếu không dùng
            if e.type == p.QUIT:
                p.quit()
                sys.exit()


            # Trong hàm run_game, vòng lặp xử lý sự kiện

            elif e.type == p.MOUSEBUTTONDOWN:

                location = p.mouse.get_pos()  # Lấy vị trí click

                # --- Xử lý click trên bàn cờ (Chỉ khi game chưa kết thúc) ---

                if not game_over and 0 <= location[0] < BOARD_WIDTH and 0 <= location[1] < BOARD_HEIGHT:

                    # Chỉ xử lý click BÀN CỜ khi game chưa over

                    col = location[0] // SQUARE_SIZE

                    row = location[1] // SQUARE_SIZE

                    if square_selected == (row, col):  # Bỏ chọn

                        square_selected = ()

                        player_clicks = []

                    else:

                        square_selected = (row, col)

                        player_clicks.append(square_selected)

                    if len(player_clicks) == 2 and human_turn:

                        move = ChessEngine.Move(player_clicks[0], player_clicks[1], game_state.board)

                        move_found = False

                        for i in range(len(valid_moves)):

                            if move == valid_moves[i]:
                                game_state.makeMove(valid_moves[i])

                                move_made = True

                                animate = True

                                square_selected = ()

                                player_clicks = []

                                move_found = True

                                break

                        if not move_found:
                            player_clicks = [square_selected]

                        # Không cần kiểm tra game_over ở đây nữa, sẽ kiểm tra ở cuối vòng lặp chính


                # --- Xử lý click nút (Luôn kiểm tra, bất kể game_over) ---

                # Chỉ xử lý click chuột trái cho các nút

                elif e.button == 1:

                    # Lấy Rect của các nút (cần có các hàm này hoặc tính toán vị trí)

                    # Ví dụ: Giả sử bạn có hàm get...Rect() trả về Rect của nút

                    back_button_rect = drawBackButton(screen, font, game_state)
                    reset_button_rect = drawResetButton(screen, font, game_state)
                    surrender_button_rect = drawSurrenderButton(screen, font, game_state)
                    return_button_rect = drawReturnButton(screen, font, game_state)

                    # Kiểm tra va chạm với các nút

                    if back_button_rect.collidepoint(location):

                        # Chỉ cho phép undo nếu game chưa kết thúc? (Thường là vậy)

                        if not game_over:

                            print("Undo button clicked (while game not over)")

                            game_state.undoMove()

                            move_made = True  # Để tính lại valid_moves

                            animate = False

                            # game_over = False # Undo không làm game hết kết thúc trừ khi bạn muốn logic đó

                            if ai_thinking:

                                if move_finder_process and move_finder_process.is_alive():
                                    move_finder_process.terminate()

                                ai_thinking = False

                            move_undone = True

                            valid_moves = game_state.getValidMoves()

                            end_game_text = ""  # Xóa text kết thúc nếu có

                        else:

                            print("Cannot undo: Game is over.")


                    elif reset_button_rect.collidepoint(location):

                        print("Reset button clicked.")

                        # Reset luôn được phép

                        game_state = ChessEngine.GameState()  # Reset trạng thái

                        valid_moves = game_state.getValidMoves()

                        square_selected = ()

                        player_clicks = []

                        move_made = False

                        animate = False

                        game_over = False  # QUAN TRỌNG: Reset cờ này

                        ai_thinking = False  # Dừng AI nếu đang nghĩ

                        if move_finder_process and move_finder_process.is_alive():
                            move_finder_process.terminate()

                        move_undone = True  # Ngăn AI đi ngay

                        end_game_text = ""  # Xóa text kết thúc


                    elif surrender_button_rect.collidepoint(location):

                        # Chỉ cho phép đầu hàng nếu game chưa kết thúc? (Hợp lý)

                        if not game_over:

                            print("Surrender button clicked.")

                            game_over = True

                            winner = "Black" if game_state.white_to_move else "White"

                            end_game_text = f"{winner} wins by Surrender"

                            # Dừng AI nếu đang nghĩ

                            if ai_thinking:

                                if move_finder_process and move_finder_process.is_alive():
                                    move_finder_process.terminate()

                                ai_thinking = False

                        else:

                            print("Cannot surrender: Game is already over.")



                    elif return_button_rect.collidepoint(location):

                        print("Return button clicked.")

                        # Return luôn được phép

                        running = False  # Thoát khỏi vòng lặp game

                        # Dừng tiến trình AI nếu đang chạy

                        if ai_thinking:

                            if move_finder_process and move_finder_process.is_alive():
                                move_finder_process.terminate()

                            ai_thinking = False


        # --- Logic AI ---
        is_ai_turn = not game_over and not human_turn and not move_undone and (player_one != player_two) # Kiểm tra lượt AI đơn giản hơn

        if is_ai_turn:
            if not ai_thinking:
                print("AI thinking...")
                ai_thinking = True
                return_queue = multiprocessing.Queue()
                ai_is_maximising = False
                # Truyền đúng ai_level đã được xác định trong main_loop
                move_finder_process = multiprocessing.Process(target=Chess_AI2.find_best_move_iddfs,
                                              args=(game_state, ai_level, return_queue, ai_is_maximising))
                move_finder_process.start()

            # Chỉ kiểm tra nếu tiến trình AI đã được khởi tạo
            if ai_thinking and move_finder_process and not move_finder_process.is_alive():
                ai_move = return_queue.get()
                if ai_move is None:
                    print("AI trả về None (hoặc lỗi), tìm nước đi ngẫu nhiên.")
                    ai_move = Chess_AI2.findRandomMove(valid_moves) # Sử dụng valid_moves hiện tại

                if ai_move: # Đảm bảo thực sự tìm/trả về được nước đi
                    game_state.makeMove(ai_move)
                    move_made = True
                    animate = True
                else:
                    print("Lỗi: AI không thể tìm thấy bất kỳ nước đi nào.")
                    # Xử lý trường hợp này - có thể là hòa cờ hoặc AI thua?
                    game_over = True
                    end_game_text = "Lỗi: AI không thể đi"

                ai_thinking = False # Đặt lại cờ


        # --- Cập nhật trạng thái game sau nước đi ---
        if move_made:
            if animate:
                animateMove(game_state.move_log[-1], screen, game_state.board, clock)
            valid_moves = game_state.getValidMoves()
            move_made = False
            animate = False
            move_undone = False # Đặt lại cờ undo

        # --- Vẽ đồ họa ---
        drawGameState(screen, game_state, valid_moves, square_selected)
        drawMoveLog(screen, game_state, move_log_font) # Sử dụng font đã truyền
        drawCustomPanel(screen, font) # Sử dụng font đã truyền
        # Vẽ các nút (cũng cần cho việc kiểm tra va chạm ở frame sau)
        back_button_rect = drawBackButton(screen, font, game_state)
        reset_button_rect = drawResetButton(screen, font, game_state)
        surrender_button_rect = drawSurrenderButton(screen, font, game_state)
        return_button_rect = drawReturnButton(screen, font, game_state)

        # --- Kiểm tra Game Over và hiển thị Text ---
        if not game_over: # Chỉ kiểm tra nếu chưa kết thúc
             if game_state.checkmate:
                 game_over = True
                 end_game_text = ("Black" if game_state.white_to_move else "White") + " win by Checkmate"
             elif game_state.stalemate:
                 game_over = True
                 end_game_text = "Stalemate"

        if game_over:
             drawEndGameText(screen, end_game_text)

        # --- Cập nhật màn hình và tick clock ---
        p.display.flip()
        clock.tick(MAX_FPS) # Sử dụng hằng số MAX_FPS của bạn

    # --- Kết thúc run_game ---
    print("Thoát khỏi phiên chơi.")


def main_loop():
    """Xử lý menu và bắt đầu các phiên chơi."""
    # --- Thiết lập Pygame và tài nguyên một lần ---
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT)) # Sử dụng hằng số
    p.display.set_caption("Chess")
    clock = p.time.Clock()
    # Tải font một lần
    font = p.font.SysFont("Verdana", 23) # Font bạn đã định nghĩa
    move_log_font = p.font.SysFont("Arial", 14, False, False) # Font bạn đã định nghĩa
    loadImages() # Tải hình ảnh một lần

    try:
        wood_tex = p.image.load("assets/wood_texture.png").convert()
        wood_tex = p.transform.scale(wood_tex, (BUTTON_WIDTH, BUTTON_HEIGHT))
        background = p.image.load("images/background.jpg").convert()  # Thêm convert()
        background = p.transform.scale(background, (WIDTH, HEIGHT))
    except p.error as e:
        print(f"Lỗi tải ảnh nền/nút: {e}")
        # Xử lý lỗi nếu cần, ví dụ tạo màu nền mặc định
        wood_tex = p.Surface((BUTTON_WIDTH, BUTTON_HEIGHT));
        wood_tex.fill(GREY)
        background = p.Surface((WIDTH, HEIGHT));
        background.fill(BLACK)

    loadImages()

    # Định nghĩa vị trí nút ở đây để truyền vào hàm menu
    buttons_main = {
        "Start Game": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 4 - BUTTON_HEIGHT // 2 + 20),
        "Quit": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 2 + 30 - BUTTON_HEIGHT // 2)
    }
    buttons_game_mode = {
        "Player vs Player": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 4 - BUTTON_HEIGHT // 2 + 20),
        "Player vs AI": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 2 + 30 - BUTTON_HEIGHT // 2),
    }
    buttons_ai_level = {
        "Easy": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 4 - BUTTON_HEIGHT // 2 + 20),
        "Hard": (WIDTH // 2 - BUTTON_WIDTH // 2 - 200, HEIGHT // 2 + 30 - BUTTON_HEIGHT // 2),
    }

    while True:
        # Giả định các hàm menu sử dụng screen, main_font đúng cách
        result = main_menu(screen, font, background, wood_tex, buttons_main)
        if result == "start_game":
            mode_result = game_mode_menu(screen, font, background, wood_tex, buttons_game_mode)

            player_one = True # Mặc định: Trắng là người
            player_two = True # Mặc định: Đen là người
            ai_level = None

            if mode_result == "player_vs_ai":
                player_two = False # Đen là AI
                ai_level = ai_level_menu(screen, font, background, wood_tex, buttons_ai_level)
                print(f"Đã chọn độ khó AI: {ai_level}")
            elif mode_result == "player_vs_player":
                print("Đã chọn chế độ Người vs Người")
            else: # Xử lý trường hợp trả về không mong muốn từ menu
                 continue # Quay lại main menu

            # --- Bắt đầu game thực tế ---
            run_game(screen, clock, font, move_log_font, player_one, player_two, ai_level)
            # --- Sau khi run_game kết thúc (vd: nhấn "Return"), vòng lặp tiếp tục về main_menu ---

        elif result == "quit":
            break # Thoát khỏi vòng lặp chính

    p.quit()
    sys.exit()

def drawGameState(screen, game_state, valid_moves, square_selected):
    """
    Responsible for all the graphics within current game state.
    """
    drawBoard(screen)  # draw squares on the board
    highlightSquares(screen, game_state, valid_moves, square_selected)
    drawPieces(screen, game_state.board)  # draw pieces on top of those squares


def drawBoard(screen):
    """
    Draw the squares on the board.
    The top left square is always light.
    """
    global colors
    colors = [p.Color(240,217,181,255), p.Color(181,136,99,255)]
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            color = colors[((row + column) % 2)]
            p.draw.rect(screen, color, p.Rect(column * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def highlightSquares(screen, game_state, valid_moves, square_selected):
    """
    Highlight the square selected and draw a small translucent circle
    on each square that is a valid destination.
    """
    if len(game_state.move_log) > 0:
        last_move = game_state.move_log[-1]
        s = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
        s.set_alpha(100)
        s.fill(p.Color('green'))
        screen.blit(s, (last_move.end_col * SQUARE_SIZE, last_move.end_row * SQUARE_SIZE))


    if square_selected != ():
        row, col = square_selected

        if game_state.board[row][col][0] == ('w' if game_state.white_to_move else 'b'):

            s = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(100)
            s.fill(p.Color(192,204,68))
            screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))

            for move in valid_moves:
                if move.start_row == row and move.start_col == col:
                    circle_surf = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
                    radius = SQUARE_SIZE // 8
                    center = (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
                    p.draw.circle(circle_surf, (32,32,32,80), center, radius)
                    screen.blit(circle_surf, (move.end_col * SQUARE_SIZE, move.end_row * SQUARE_SIZE))


def drawPieces(screen, board):
    """
    Draw the pieces on the board using the current game_state.board
    """
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            piece = board[row][column]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(column * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def drawMoveLog(screen, game_state, move_log_font):
    """
    Draws the move log.
    """
    global scroll_y
    move_log_rect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color(40, 36, 36), move_log_rect)
    move_log = game_state.move_log
    move_texts = []

    for i in range(0, len(move_log), 2):
        move_string = f"{i // 2 + 1}. {move_log[i]}  "
        if i + 1 < len(move_log):
            move_string += f"{move_log[i + 1]}     "
        move_texts.append(move_string)

    moves_per_row = 3
    padding = 10
    line_spacing = 6
    text_y = padding - scroll_y


    for i in range(0, len(move_texts), moves_per_row):
        text = "".join(move_texts[i:i + moves_per_row])
        text_object = move_log_font.render(text, True, p.Color('white'))
        text_location = move_log_rect.move(padding, text_y)
        screen.blit(text_object, text_location)
        text_y += text_object.get_height() + line_spacing

def handle_scroll(event):

    global scroll_y
    if event.type == p.MOUSEBUTTONDOWN:
        if event.button == 4:# Scroll up
            scroll_y = max(0, scroll_y - 20)
        elif event.button == 5: # Scroll down
            scroll_y += 20


def drawCustomPanel(screen, font):
    custom_panel_rect = p.Rect(BOARD_WIDTH, MOVE_LOG_PANEL_HEIGHT, MOVE_LOG_PANEL_WIDTH, HEIGHT - MOVE_LOG_PANEL_HEIGHT)  # Chúng ta vẽ panel tại vị trí khuyết
    p.draw.rect(screen, p.Color(64,60,60), custom_panel_rect)

def drawBackButton(screen, font, game_state):
    back_button_rect = p.Rect(BOARD_WIDTH + 22, MOVE_LOG_PANEL_HEIGHT + 25, 100, 50)
    back_button_image = p.image.load("images/back.png")
    back_button_image = p.transform.smoothscale(back_button_image, (70, 70))
    screen.blit(back_button_image, back_button_rect.topleft)
    return back_button_rect

def drawResetButton(screen, font, game_state):
    reset_button_rect = p.Rect(BOARD_WIDTH + 92, MOVE_LOG_PANEL_HEIGHT + 25, 100, 50)
    reset_button_image = p.image.load("images/reset.png")
    reset_button_image = p.transform.smoothscale(reset_button_image, (70, 70))
    screen.blit(reset_button_image, reset_button_rect.topleft)
    return reset_button_rect

def drawSurrenderButton(screen, font, game_state):
    surrender_button_rect = p.Rect(BOARD_WIDTH + 162, MOVE_LOG_PANEL_HEIGHT + 25, 100, 50)
    surrender_button_image = p.image.load("images/surrender.png")
    surrender_button_image = p.transform.smoothscale(surrender_button_image, (70, 70))
    screen.blit(surrender_button_image, surrender_button_rect.topleft)
    return surrender_button_rect


def drawReturnButton(screen, font, game_state):
    return_button_rect = p.Rect(BOARD_WIDTH + 230, MOVE_LOG_PANEL_HEIGHT + 110, 100, 50)
    return_button_image = p.image.load("images/return.png")
    return_button_image = p.transform.smoothscale(return_button_image, (20, 20))
    screen.blit(return_button_image, return_button_rect.topleft)
    return return_button_rect


def drawEndGameText(screen, text):
    font = p.font.SysFont("Helvetica", 32, True, False)
    text_object = font.render(text, False, p.Color("gray"))
    text_location = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(BOARD_WIDTH / 2 - text_object.get_width() / 2,
                                                                 BOARD_HEIGHT / 2 - text_object.get_height() / 2)
    screen.blit(text_object, text_location)
    text_object = font.render(text, False, p.Color('black'))
    screen.blit(text_object, text_location.move(2, 2))


def animateMove(move, screen, board, clock):
    """
    Animating a move
    """
    global colors
    d_row = move.end_row - move.start_row
    d_col = move.end_col - move.start_col
    frames_per_square = 10  # frames to move one square
    frame_count = (abs(d_row) + abs(d_col)) * frames_per_square
    for frame in range(frame_count + 1):
        row, col = (move.start_row + d_row * frame / frame_count, move.start_col + d_col * frame / frame_count)
        drawBoard(screen)
        drawPieces(screen, board)
        # erase the piece moved from its ending square
        color = colors[(move.end_row + move.end_col) % 2]
        end_square = p.Rect(move.end_col * SQUARE_SIZE, move.end_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        p.draw.rect(screen, color, end_square)
        # draw captured piece onto rectangle
        if move.piece_captured != '--':
            if move.is_enpassant_move:
                enpassant_row = move.end_row + 1 if move.piece_captured[0] == 'b' else move.end_row - 1
                end_square = p.Rect(move.end_col * SQUARE_SIZE, enpassant_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            screen.blit(IMAGES[move.piece_captured], end_square)
        # draw moving piece
        screen.blit(IMAGES[move.piece_moved], p.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
        p.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    # main()
    freeze_support()
    main_loop()
