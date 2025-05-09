# zobrist.py
import random

PIECE_TYPES = 12
SQUARES = 64

# QUAN TRỌNG: Sử dụng ký hiệu quân nhất quán với GameState.board
# Nếu GameState dùng 'wp', 'bp', thì map này cũng phải dùng 'wp', 'bp'
zobrist_piece_map = {
    'wp': 0, 'wN': 1, 'wB': 2, 'wR': 3, 'wQ': 4, 'wK': 5,
    'bp': 6, 'bN': 7, 'bB': 8, 'bR': 9, 'bQ': 10, 'bK': 11
}
zobrist_idx_to_piece = {v: k for k, v in zobrist_piece_map.items()}

zobrist_table = [[0] * PIECE_TYPES for _ in range(SQUARES)]
zobrist_black_to_move = 0
zobrist_castling_rights = [0] * 4  # Index 0:wks, 1:wqs, 2:bks, 3:bqs
zobrist_en_passant_file = [0] * 8   # Index 0:file a, ..., 7:file h

def init_zobrist():
    global zobrist_black_to_move, zobrist_castling_rights, zobrist_en_passant_file

    for i in range(SQUARES):
        for j in range(PIECE_TYPES):
            zobrist_table[i][j] = random.getrandbits(64)
    zobrist_black_to_move = random.getrandbits(64)
    for i in range(4):
        zobrist_castling_rights[i] = random.getrandbits(64)
    for i in range(8):
        zobrist_en_passant_file[i] = random.getrandbits(64)

init_zobrist()

def get_zobrist_piece_index(piece_string):

    return zobrist_piece_map.get(piece_string)