import chess
import chess.engine
from ChessAI import findBestMoveFromModel
from ChessEngine import GameState
import math

stockfish_path = r"D:\KieuQuy\Documents\AI\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

def fen_to_gamestate(board):
    gs = GameState()
    piece_map = board.piece_map()
    board_array = [["--" for _ in range(8)] for _ in range(8)]

    for square, piece in piece_map.items():
        row = 7 - square // 8
        col = square % 8
        color = 'w' if piece.color == chess.WHITE else 'b'
        board_array[row][col] = color + piece.symbol().upper()

    gs.board = board_array
    gs.white_to_move = board.turn == chess.WHITE
    return gs

def move_to_uci(move, board):
    uci = f"{chr(move.start_col+97)}{8-move.start_row}{chr(move.end_col+97)}{8-move.end_row}"
    uci_move = chess.Move.from_uci(uci)
    if uci_move in board.legal_moves:
        return uci_move
    else:
        raise ValueError(f"AI chọn nước không hợp lệ: {uci} ở vị trí FEN: {board.fen()}")

def play_game(ai_color="white", stockfish_level=3):
    board = chess.Board()
    limit = chess.engine.Limit(depth=10)

    while not board.is_game_over():
        try:
            if (board.turn == chess.WHITE and ai_color == "white") or (board.turn == chess.BLACK and ai_color == "black"):
                game_state = fen_to_gamestate(board)
                valid_moves = game_state.getValidMoves()
                ai_move = findBestMoveFromModel(game_state, valid_moves)
                board.push(move_to_uci(ai_move, board))
            else:
                result = engine.play(board, limit)
                board.push(result.move)
        except Exception as e:
            raise RuntimeError(f"Lỗi khi xử lý nước đi: {e}")

    outcome = board.outcome()
    if outcome.winner is None:
        return 0.5
    elif (outcome.winner and ai_color == "white") or (not outcome.winner and ai_color == "black"):
        return 1
    else:
        return 0

def estimate_elo(R_opponent, S_a):
    if S_a <= 0:
        return R_opponent - 800
    elif S_a >= 1:
        return R_opponent + 800
    return round(R_opponent - 400 * math.log10((1 - S_a) / S_a))

total_games = 50
ai_color = "white"
stockfish_elo = 1350
stockfish_level = 3

score = 0
for i in range(total_games):
    print(f"\n🔹 Đang chơi ván {i+1}/{total_games}...")
    try:
        result = play_game(ai_color=ai_color, stockfish_level=stockfish_level)
        score += result
        print(f"Kết quả: {'Thắng' if result==1 else 'Hòa' if result==0.5 else 'Thua'}")
    except Exception as e:
        print(f"Lỗi ở ván {i+1}: {e}")

S_a = score / total_games
estimated_elo = estimate_elo(stockfish_elo, S_a)

print(f"\nAI thắng trung bình: {S_a*100:.1f}%")
print(f"Ước lượng ELO của AI: {estimated_elo}")

engine.quit()
