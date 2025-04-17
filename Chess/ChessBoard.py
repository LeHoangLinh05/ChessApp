"""
Main driver file.
Handling user input.
Displaying current GameStatus object.
"""
import pygame
import pygame as p
import ChessEngine, ChessAI
import sys
from multiprocessing import Process, Queue

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

# Màu sắc
WHITE = p.Color("white")
BLACK = p.Color("black")
GREY = p.Color("grey")

# Khởi tạo Pygame
p.init()

# Tạo cửa sổ game
screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
p.display.set_caption("Chess")

# Phông chữ
font = p.font.SysFont("Verdana", 23)
move_log_font = p.font.SysFont("Verdana", 30, False, False)
wood_tex = p.image.load("assets/wood_texture.png").convert()
wood_tex = p.transform.scale(wood_tex, (BUTTON_WIDTH, BUTTON_HEIGHT))
background = p.image.load("images/background.jpg")
background = p.transform.scale(background, (WIDTH, HEIGHT))


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


def draw_button(screen, text, x, y):
    screen.blit(wood_tex, (x, y))
    light = (200, 180, 150)
    dark  = ( 80,  60,  30)
    p.draw.line(screen, light, (x, y), (x+BUTTON_WIDTH, y), 3)
    p.draw.line(screen, light, (x, y), (x, y+BUTTON_HEIGHT), 3)
    p.draw.line(screen, dark, (x, y+BUTTON_HEIGHT), (x+BUTTON_WIDTH, y+BUTTON_HEIGHT), 3)
    p.draw.line(screen, dark, (x+BUTTON_WIDTH, y), (x+BUTTON_WIDTH, y+BUTTON_HEIGHT), 3)

    text_surf = font.render(text, True, (250, 250, 247))
    shadow = font.render(text, True, (0, 0, 0))
    tx, ty = x + BUTTON_WIDTH//2, y + BUTTON_HEIGHT//2

    shadow_rect = shadow.get_rect(center=(tx+2, ty+2))
    screen.blit(shadow, shadow_rect)
    text_rect = text_surf.get_rect(center=(tx, ty))
    screen.blit(text_surf, text_rect)

def main_menu():
    while True:
        screen.fill(WHITE)
        screen.blit(background, (0, 0))

        for button_text, (x, y) in buttons_main.items():
            draw_button(screen,button_text, x, y)

        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos

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


def game_mode_menu():

    while True:
        screen.fill(WHITE)
        screen.blit(background, (0, 0))

        for button_text, (x, y) in buttons_game_mode.items():
            draw_button(screen,button_text, x, y)

        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos

                if buttons_game_mode["Player vs Player"][0] <= mouse_x <= buttons_game_mode["Player vs Player"][
                    0] + BUTTON_WIDTH and buttons_game_mode["Player vs Player"][1] <= mouse_y <= \
                        buttons_game_mode["Player vs Player"][1] + BUTTON_HEIGHT:
                    print("Player vs Player mode")
                    return "player_vs_player"
                elif buttons_game_mode["Player vs AI"][0] <= mouse_x <= buttons_game_mode["Player vs AI"][
                    0] + BUTTON_WIDTH and buttons_game_mode["Player vs AI"][1] <= mouse_y <= \
                        buttons_game_mode["Player vs AI"][1] + BUTTON_HEIGHT:
                    print("Player vs AI mode")
                    return "player_vs_ai"

            p.display.flip()


def ai_level_menu():

    while True:
        screen.fill(WHITE)
        screen.blit(background, (0, 0))

        for button_text, (x, y) in buttons_ai_level.items():
            draw_button(screen,button_text, x, y)

        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                sys.exit()
            elif event.type == p.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos

                if buttons_ai_level["Easy"][0] <= mouse_x <= buttons_ai_level["Easy"][0] + BUTTON_WIDTH and \
                        buttons_ai_level["Easy"][1] <= mouse_y <= buttons_ai_level["Easy"][1] + BUTTON_HEIGHT:
                    print("AI level: Easy")
                    return "easy"
                elif buttons_ai_level["Hard"][0] <= mouse_x <= buttons_ai_level["Hard"][0] + BUTTON_WIDTH and \
                        buttons_ai_level["Hard"][1] <= mouse_y <= buttons_ai_level["Hard"][1] + BUTTON_HEIGHT:
                    print("AI level: Hard")
                    return "hard"

            p.display.flip()

def loadImages():
    """
    Initialize a global directory of images.
    This will be called exactly once in the main.
    """
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQUARE_SIZE, SQUARE_SIZE))


def main():
    # p.init()
    # screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
    clock = p.time.Clock()
    font = p.font.SysFont("Arial", 14, False, False)

    while True:
        result = main_menu()
        if result == "start_game":
            mode_result = game_mode_menu()

            if mode_result == "player_vs_ai":
                ai_level = ai_level_menu()
                print(f"AI level selected: {ai_level}")
                player_one = True
                player_two = False
            else:
                print("Player vs Player mode selected")
                player_one = True
                player_two = True

            # Khởi tạo trò chơi
            game_state = ChessEngine.GameState()
            valid_moves = game_state.getValidMoves()
            move_made = False
            animate = False
            loadImages()

            square_selected = ()
            player_clicks = []
            game_over = False
            ai_thinking = False
            move_undone = False
            move_finder_process = None
            move_log_font = p.font.SysFont("Arial", 14, False, False)

            running = True
            while running:
                human_turn = (game_state.white_to_move and player_one) or (not game_state.white_to_move and player_two)

                for e in p.event.get():
                    if e.type == p.QUIT:
                        p.quit()
                        sys.exit()

                    elif e.type == p.MOUSEBUTTONDOWN:
                        if not game_over:
                            location = p.mouse.get_pos()
                            col = location[0] // SQUARE_SIZE
                            row = location[1] // SQUARE_SIZE

                            if square_selected == (row, col) or col >= 8:
                                square_selected = ()
                                player_clicks = []
                            else:
                                square_selected = (row, col)
                                player_clicks.append(square_selected)

                            if len(player_clicks) == 2 and human_turn:
                                move = ChessEngine.Move(player_clicks[0], player_clicks[1], game_state.board)
                                for i in range(len(valid_moves)):
                                    if move == valid_moves[i]:
                                        game_state.makeMove(valid_moves[i])
                                        move_made = True
                                        animate = True
                                        square_selected = ()
                                        player_clicks = []
                                if not move_made:
                                    player_clicks = [square_selected]

                        if e.button == 1:
                            mouse_pos = p.mouse.get_pos()
                            back_button_rect = drawBackButton(screen, font, game_state)
                            reset_button_rect = drawResetButton(screen, font, game_state)
                            surrender_button_rect = drawSurrenderButton(screen, font, game_state)
                            return_button_rect = drawReturnButton(screen, font, game_state)

                            if back_button_rect.collidepoint(mouse_pos):
                                game_state.undoMove()
                                move_made = True
                                animate = False
                                game_over = False
                                if ai_thinking:
                                    move_finder_process.terminate()
                                    ai_thinking = False
                                move_undone = True

                            elif reset_button_rect.collidepoint(mouse_pos):
                                game_state = ChessEngine.GameState()
                                valid_moves = game_state.getValidMoves()
                                square_selected = ()
                                player_clicks = []
                                move_made = False
                                animate = False
                                game_over = False
                                if ai_thinking:
                                    move_finder_process.terminate()
                                    ai_thinking = False
                                move_undone = True

                            elif surrender_button_rect.collidepoint(mouse_pos):
                                game_over = True
                                end_game_text = "Opponent wins by resign"

                            elif return_button_rect.collidepoint(mouse_pos):
                                running = False

                # AI move finder
                if not game_over and not human_turn and not move_undone and mode_result == "player_vs_ai":
                    if not ai_thinking:
                        ai_thinking = True
                        return_queue = Queue()
                        move_finder_process = Process(target=ChessAI.findBestMove,
                                                      args=(game_state, valid_moves, ai_level, return_queue))
                        move_finder_process.start()

                    if not move_finder_process.is_alive():
                        ai_move = return_queue.get()
                        if ai_move is None:
                            ai_move = ChessAI.findRandomMove(valid_moves)
                        game_state.makeMove(ai_move)
                        move_made = True
                        animate = True
                        ai_thinking = False

                if move_made:
                    if animate:
                        animateMove(game_state.move_log[-1], screen, game_state.board, clock)
                    valid_moves = game_state.getValidMoves()
                    move_made = False
                    animate = False
                    move_undone = False

                drawGameState(screen, game_state, valid_moves, square_selected)

                if not game_over:
                    drawMoveLog(screen, game_state, move_log_font)
                    drawCustomPanel(screen, font)
                    back_button_rect = drawBackButton(screen, font, game_state)
                    reset_button_rect = drawResetButton(screen, font, game_state)
                    surrender_button_rect = drawSurrenderButton(screen, font, game_state)
                    return_button_rect = drawReturnButton(screen, font, game_state)

                if game_state.checkmate:
                    game_over = True
                    txt = ("Black" if game_state.white_to_move else "White") + " wins by checkmate"
                    drawEndGameText(screen, txt)
                elif game_state.stalemate:
                    game_over = True
                    drawEndGameText(screen, "Stalemate")
                elif game_over:
                    drawEndGameText(screen, end_game_text)

                clock.tick(15)
                p.display.flip()

        elif result == "quit":
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
    move_log_rect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color(40,36,36), move_log_rect)
    move_log = game_state.move_log
    move_texts = []
    for i in range(0, len(move_log), 2):
        move_string = str(i // 2 + 1) + '. ' + str(move_log[i]) + "  "
        if i + 1 < len(move_log):
            move_string += str(move_log[i + 1]) + "     "
        move_texts.append(move_string)

    moves_per_row = 3
    padding = 10
    line_spacing = 6
    text_y = padding
    for i in range(0, len(move_texts), moves_per_row):
        text = ""
        for j in range(moves_per_row):
            if i + j < len(move_texts):
                text += move_texts[i + j]

        text_object = move_log_font.render(text, True, p.Color('white'))
        text_location = move_log_rect.move(padding, text_y)
        screen.blit(text_object, text_location)
        text_y += text_object.get_height() + line_spacing

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
    main()
