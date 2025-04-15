import pygame
import sys
from pygame.locals import *
from PIL import Image

# Khởi tạo pygame
pygame.init()

# Kích thước cửa sổ và các ô
WIDTH = 600
HEIGHT = 600
SQUARE_SIZE = WIDTH // 8
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 100

# Kích thước của sidebar
SIDEBAR_WIDTH = 250
HISTORY_BOX_HEIGHT = HEIGHT - 100  # Chiều cao khung lịch sử

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)  # Màu nâu cho ô đen
LIGHT_BROWN = (255, 222, 185)  # Màu sáng cho ô trắng
LIGHT_GREY = (200, 200, 200)

# Tạo cửa sổ
screen = pygame.display.set_mode((WIDTH + SIDEBAR_WIDTH, HEIGHT))  # Thêm không gian cho sidebar
pygame.display.set_caption("Cờ Vua")

# Biến cuộn
scroll_offset = 0

# Tải hình ảnh quân cờ
def load_piece_images():
    piece_images = {}
    pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    colors = ["white", "black"]

    for color in colors:
        for piece in pieces:
            img_path = f"images/{color}_{piece}.png"
            try:
                # Mở hình ảnh bằng Pillow
                img = Image.open(img_path)

                # Điều chỉnh kích thước sao cho vừa với ô cờ mà không bị mờ
                img = img.resize((SQUARE_SIZE - 20, SQUARE_SIZE - 20),
                                 Image.Resampling.LANCZOS)  # Sử dụng LANCZOS thay cho ANTIALIAS

                # Chuyển đổi hình ảnh Pillow sang Pygame surface
                img = pygame.image.fromstring(img.tobytes(), img.size, img.mode)

                # Lưu lại hình ảnh quân cờ đã được chỉnh kích thước
                piece_images[f"{color}_{piece}"] = img
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return piece_images

# Vẽ bàn cờ
def draw_board():
    for row in range(8):
        for col in range(8):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Vẽ quân cờ
def draw_pieces(piece_images, board):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != ' ':
                piece_image = piece_images.get(piece)
                if piece_image:
                    # Tính toán vị trí căn giữa của quân cờ trong ô
                    x_pos = col * SQUARE_SIZE + (SQUARE_SIZE - piece_image.get_width()) // 2
                    y_pos = row * SQUARE_SIZE + (SQUARE_SIZE - piece_image.get_height()) // 2

                    # Vẽ quân cờ vào đúng vị trí
                    screen.blit(piece_image, (x_pos, y_pos))


# Vẽ sidebar (lịch sử các nước cờ và các nút điều khiển)
def draw_sidebar():
    pygame.draw.rect(screen, LIGHT_GREY, (WIDTH, 0, SIDEBAR_WIDTH, HEIGHT))  # Sidebar nền xám

    # Hiển thị tiêu đề của lịch sử nước cờ
    history_font = pygame.font.SysFont("Arial", 18)
    history_title = history_font.render("Lịch sử Nước Cờ", True, BLACK)
    screen.blit(history_title, (WIDTH + 10, 10))  # Tiêu đề lịch sử nước cờ

    # Vẽ khung lịch sử nước cờ
    pygame.draw.rect(screen, WHITE, (WIDTH + 10, 50, SIDEBAR_WIDTH - 20, HISTORY_BOX_HEIGHT))
    pygame.draw.rect(screen, BLACK, (WIDTH + 10, 50, SIDEBAR_WIDTH - 20, HISTORY_BOX_HEIGHT), 2)  # Viền khung

    # Danh sách các nước cờ (ví dụ)
    move_history = ["1. f4", "2. e4", "3. c6", "4. a5", "5. Nf3", "6. d5", "7. e6", "8. Bf4", "9. e5", "10. Bxe5",
                    "11. Bxe5", "12. Bxe5"]  # Dữ liệu ví dụ dài

    # Hiển thị lịch sử các nước cờ
    history_font = pygame.font.SysFont("Arial", 16)
    y_offset = 55 - scroll_offset  # Áp dụng cuộn vào vị trí

    for i, move in enumerate(move_history):
        move_text = history_font.render(move, True, BLACK)
        screen.blit(move_text, (WIDTH + 15, y_offset + (i * 30)))  # Vị trí của từng nước cờ

        # Nếu danh sách quá dài, cho phép cuộn
        if y_offset + (i * 30) > HISTORY_BOX_HEIGHT:
            break

def drawEndGameText(screen, text):
    font = pygame.font.SysFont("Helvetica", 32, True, False)
    text_object = font.render(text, False, pygame.Color("gray"))
    text_location = pygame.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH / 2 - text_object.get_width() / 2,
                                                                 HEIGHT / 2 - text_object.get_height() / 2)
    screen.blit(text_object, text_location)
    text_object = font.render(text, False, pygame.Color('black'))
    screen.blit(text_object, text_location.move(2, 2))

# Vẽ các nút điều khiển trong sidebar
def draw_buttons():
    font = pygame.font.SysFont("Arial", 18)
    buttons = ["Tua lại", "Tua đi", "Đầu hàng"]  # Các nút điều khiển

    for i, button in enumerate(buttons):
        pygame.draw.rect(screen, WHITE, (WIDTH + 75, 150 + i * 60, 100, 40))  # Vẽ nền nút
        pygame.draw.rect(screen, BLACK, (WIDTH + 75, 150 + i * 60, 100, 40), 2)  # Viền nút
        text = font.render(button, True, BLACK)  # Văn bản của nút
        screen.blit(text, (WIDTH + 85, 160 + i * 60))  # Hiển thị văn bản trên nút

# Khởi tạo bàn cờ và các quân cờ
def initialize_board():
    board = [
        ["black_rook", "black_knight", "black_bishop", "black_queen", "black_king", "black_bishop", "black_knight", "black_rook"],
        ["black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn", "black_pawn"],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        ["white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn", "white_pawn"],
        ["white_rook", "white_knight", "white_bishop", "white_queen", "white_king", "white_bishop", "white_knight", "white_rook"]
    ]
    return board

# Xử lý di chuyển quân cờ
# def move_piece(start_pos, end_pos, board):
#     start_row, start_col = start_pos
#     end_row, end_col = end_pos
#
#     # Kiểm tra xem các chỉ số có hợp lệ không (nằm trong phạm vi 8x8)
#     if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
#         print("Lỗi: Chỉ số không hợp lệ!")
#         return  # Trả về nếu chỉ số không hợp lệ
#
#     piece = board[start_row][start_col]
#     if piece == " ":
#         print("Lỗi: Không có quân cờ để di chuyển!")
#         return  # Nếu không có quân cờ ở ô bắt đầu
#
#     # Di chuyển quân cờ
#     board[start_row][start_col] = " "
#     board[end_row][end_col] = piece
#
#     # Cập nhật lịch sử nước cờ
#     move_history.append(f"{piece} từ {chr(97 + start_col)}{8 - start_row} đến {chr(97 + end_col)}{8 - end_row}")

# Hàm chính để chạy chương trình
def main():
    global scroll_offset
    # Tải hình ảnh quân cờ
    piece_images = load_piece_images()

    # Cập nhật trạng thái bàn cờ
    board = initialize_board()

    selected_piece = None  # Biến lưu quân cờ đang được chọn

    running = True
    while running:
        screen.fill(WHITE)  # Lấp đầy màn hình với màu trắng

        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Lắng nghe sự kiện nhấp chuột để chọn quân cờ
            if event.type == pygame.MOUSEBUTTONDOWN:
                col = event.pos[0] // SQUARE_SIZE
                row = event.pos[1] // SQUARE_SIZE

                if selected_piece is None:
                    # Nếu chưa chọn quân cờ, chọn quân cờ ở vị trí hiện tại
                    selected_piece = (row, col)
                else:
                    # Di chuyển quân cờ nếu đã chọn
                    move_piece(selected_piece, (row, col), board)
                    selected_piece = None

            # Xử lý sự kiện cuộn chuột
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Cuộn lên
                    scroll_offset = max(0, scroll_offset - 30)  # Giới hạn cuộn lên
                elif event.button == 5:  # Cuộn xuống
                    scroll_offset += 30  # Cuộn xuống

        # Vẽ bàn cờ
        draw_board()

        # Vẽ quân cờ
        draw_pieces(piece_images, board)

        # Vẽ sidebar và các nút
        draw_sidebar()
        draw_buttons()

        # Cập nhật màn hình
        pygame.display.flip()

    # Dừng pygame khi thoát
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
