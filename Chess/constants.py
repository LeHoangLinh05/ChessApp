import math

#
# Chess constants
#

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
EMPTY = 13

A1, B1, C1, D1, E1, F1, G1, H1 = 91, 92, 93, 94, 95, 96, 97, 98
A2, H2 = 81, 88
A8, H8 = 21, 28
last_rank = [28, 27, 26, 25, 24, 23, 22, 21]

fen_pieces = 'PNBRQKpnbrqk'

# Directions for generating moves on a 10x12 board
N, E, S, W = -10, 1, 10, -1

directions = {
    PAWN: (N, N + N, N + W, N + E),
    KNIGHT: (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    BISHOP: (N + E, S + E, S + W, N + W),
    ROOK: (N, E, S, W),
    QUEEN: (N, E, S, W, N + E, S + E, S + W, N + W),
    KING: (N, E, S, W, N + E, S + E, S + W, N + W)
}
directions_isatt = {
    BISHOP + 6: (N + E, S + E, S + W, N + W),
    ROOK + 6: (N, E, S, W),
    KNIGHT + 6: (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    PAWN + 6: (N + E, N + W),
    KING + 6: (N, E, S, W, N + E, S + E, S + W, N + W)
}

# 10x12 board
mailbox = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 56, 57, 58, 59, 60, 61, 62, 63, -1,
    -1, 48, 49, 50, 51, 52, 53, 54, 55, -1,
    -1, 40, 41, 42, 43, 44, 45, 46, 47, -1,
    -1, 32, 33, 34, 35, 36, 37, 38, 39, -1,
    -1, 24, 25, 26, 27, 28, 29, 30, 31, -1,
    -1, 16, 17, 18, 19, 20, 21, 22, 23, -1,
    -1, 8, 9, 10, 11, 12, 13, 14, 15, -1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
]

mailbox64 = [
    98, 97, 96, 95, 94, 93, 92, 91,
    88, 87, 86, 85, 84, 83, 82, 81,
    78, 77, 76, 75, 74, 73, 72, 71,
    68, 67, 66, 65, 64, 63, 62, 61,
    58, 57, 56, 55, 54, 53, 52, 51,
    48, 47, 46, 45, 44, 43, 42, 41,
    38, 37, 36, 35, 34, 33, 32, 31,
    28, 27, 26, 25, 24, 23, 22, 21
]

square_name = [
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'
]

char_to_piece = {
    'K': KING,
    'Q': QUEEN,
    'R': ROOK,
    'B': BISHOP,
    'N': KNIGHT,
    'P': PAWN,
    'k': KING + 6,
    'q': QUEEN + 6,
    'r': ROOK + 6,
    'b': BISHOP + 6,
    'n': KNIGHT + 6,
    'p': PAWN + 6
}

#
# Search constants
#

# Only for ordering captures
piece_vals = {PAWN: 1, KNIGHT: 2, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}

MATE_SCORE = 200000
HIST_MAX = 10000
TT_MAX = 1e6 # Giữ nguyên tên gốc TT_MAX
TT_EXACT = 1
TT_UPPER = 2
TT_LOWER = 3
PRUNE_MARGIN = 180
RAZOR_MARGIN = 500
ASPIRATION_DELTA = 15

def lmr(d, lm): return 1 + int(math.log(d) * math.log(lm) * 0.5)

#
# Misc functions
#

def my_piece(piece):
    return piece >= 1 and piece <= 6

def opp_piece(piece):
    return piece >= 7 and piece <= 12

def is_pnk(piece): # piece là piece_type 1-6
    return piece == PAWN or piece == KNIGHT or piece == KING

def invert_sides(board):
    for i, piece in enumerate(board):
        if opp_piece(piece):
            board[i] -= 6
        elif my_piece(piece):
            board[i] += 6