import time
import math
from ctypes import *
from constants import *

# Load NNUE probe and init weights
nnue = cdll.LoadLibrary('D:/AI/Chess/ChessApp/Chess/libnnueprobe.dll')
nnue.nnue_init(b'D:/AI/Chess/ChessApp/Chess/net_epoch3.nnue')


#
# Game class and chess logic
#

class Game():
    def __init__(self, brd, s, wc, wlc, bc, blc, eps, pos):
        self.board = brd  # 10x12 board
        self.side = s  # White/Black

        self.w_castle = wc  # Castle rights
        self.w_lcastle = wlc
        self.b_castle = bc
        self.b_lcastle = blc

        self.enp = eps  # En Passant square

        self.positions = pos

    def copy(self):
        return Game(self.board.copy(), self.side, self.w_castle, self.w_lcastle, self.b_castle, self.b_lcastle,
                    self.enp, self.positions)

    def close_to_startpos(self):
        cnt, wcnt, bcnt = 0, 0, 0
        while cnt < 16:
            if my_piece(self.board[mailbox64[cnt]]): wcnt += 1
            if opp_piece(self.board[mailbox64[63 - cnt]]): bcnt += 1
            cnt += 1
        return wcnt >= 11 and bcnt >= 11

    def initial_pos(self):
        self.parse_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    def uci_move(self, move):
        move_str = square_name[mailbox[move[0]]] + square_name[mailbox[move[1]]] if self.side else \
            square_name[mailbox[119 - move[0]]] + square_name[mailbox[119 - move[1]]]
        if self.board[move[0]] == PAWN and move[1] in last_rank:
            move_str += 'q'
        return move_str

    def parse_fen(self, fen):
        self.board = [0] * 120
        self.positions = []
        self.side = True
        self.enp = 0

        fen_split = fen.split()

        idx = 63
        for c in fen_split[0]:
            if c in '12345678':
                for i in range(int(c)):
                    self.board[mailbox64[idx]] = EMPTY
                    idx -= 1
            elif c == '/':
                continue
            else:
                self.board[mailbox64[idx]] = char_to_piece[c]
                idx -= 1
        if fen_split[1] == 'b':
            self.rotate()

        self.w_castle = 'K' in fen_split[2]
        self.w_lcastle = 'Q' in fen_split[2]
        self.b_castle = 'k' in fen_split[2]
        self.b_lcastle = 'q' in fen_split[2]

        self.positions.append(self.to_fen())

    def to_fen(self):  # For NNUE evaluation
        fen = ''
        cpy = self.copy()
        if not self.side:
            cpy.rotate()
        cnt = 0
        empties = 0
        breaks = 0
        for piece in cpy.board:
            if piece == 0: continue
            if piece == EMPTY:
                empties += 1
            else:
                if empties != 0:
                    fen += str(empties)
                    empties = 0
                fen += fen_pieces[piece - 1]
            cnt += 1
            if cnt % 8 == 0:
                breaks += 1
                if empties != 0:
                    fen += str(empties)
                    empties = 0
                if breaks != 8: fen += '/'
        fen += ' w ' if self.side else ' b '
        fen += '- -'

        return fen

    def rotate(self):
        invert_sides(self.board)
        self.board.reverse()

        self.w_castle, self.w_lcastle, self.b_castle, self.b_lcastle = \
            self.b_castle, self.b_lcastle, self.w_castle, self.w_lcastle

        self.enp = 119 - self.enp
        self.side = not self.side

    def movegen(self):  # Move generation using directions
        movelist = []
        for sq, piece in enumerate(self.board):
            if not my_piece(piece): continue
            for d in directions[piece]:
                next_sq = sq
                while True:
                    next_sq += d
                    # If next square is our piece or off the board
                    if self.board[next_sq] <= 6: break

                    if piece == PAWN:
                        # Pawns cannot capture when going directly forward
                        if d == N and self.board[next_sq] != EMPTY: break
                        if d == N + N and (self.board[next_sq] != EMPTY or self.board[next_sq - N] != EMPTY \
                                           or sq < A2 or sq > H2): break
                        if (d == N + W or d == N + E) and not (
                                opp_piece(self.board[next_sq]) or self.enp == next_sq): break
                    movelist.append((sq, next_sq))
                    if opp_piece(self.board[next_sq]): break

                    # Break for a non sliding piece
                    if is_pnk(piece): break

                    # Castling rights
                    if sq == A1 and self.w_lcastle:
                        if self.board[next_sq + E] == KING and not (
                                self.is_attacked(next_sq + E) or self.is_attacked(next_sq) or self.is_attacked(
                                next_sq + W)):
                            movelist.append((next_sq + E, next_sq + W))
                    elif sq == H1 and self.w_castle:
                        if self.board[next_sq + W] == KING and not (
                                self.is_attacked(next_sq + E) or self.is_attacked(next_sq) or self.is_attacked(
                                next_sq + W)):
                            movelist.append((next_sq + W, next_sq + E))

        return movelist

    def is_attacked(self, sq):
        for piece, dirs in directions_isatt.items():
            for d in dirs:
                next_sq = sq
                while True:
                    next_sq += d
                    if self.board[next_sq] <= 6: break
                    if self.board[next_sq] == piece: return True
                    if (piece == BISHOP + 6 or piece == ROOK + 6) and self.board[next_sq] == QUEEN + 6: return True
                    if opp_piece(self.board[next_sq]): break
                    if is_pnk(piece - 6): break
        return False

    def is_capture(self, move):
        return opp_piece(self.board[move[1]])

    def in_check(self):
        for i in range(120):
            if self.board[i] == KING:
                if self.is_attacked(i): return True
        return False

    def make_move(self, move):
        fr, to = move
        enpcpy = self.enp
        self.enp = 0

        if self.w_lcastle and fr == A1:
            self.w_lcastle = False
        elif self.w_castle and fr == H1:
            self.w_castle = False

        if self.board[fr] == KING:
            self.w_castle = self.w_lcastle = False
            if fr == E1:
                if to == C1:
                    self.board[A1] = EMPTY
                    self.board[D1] = ROOK
                elif to == G1:
                    self.board[H1] = EMPTY
                    self.board[F1] = ROOK
            elif fr == D1:
                if to == B1:
                    self.board[A1] = EMPTY
                    self.board[C1] = ROOK
                elif to == F1:
                    self.board[H1] = EMPTY
                    self.board[E1] = ROOK
        self.board[to] = self.board[fr]

        if self.board[fr] == PAWN:
            if to in last_rank:
                self.board[to] = QUEEN
            if to == fr + N + N:
                self.enp = fr + N
            if to == enpcpy:
                self.board[to + S] = EMPTY

        self.board[fr] = EMPTY

    def make_uci_move(self, move):
        for m in self.movegen():
            if move == self.uci_move(m):
                self.make_move(m)
                self.rotate()
                self.positions.append(self.to_fen())
                break

    def uci_position(self, command):
        is_move = False
        for part in command.split()[1:]:
            if is_move:
                self.make_uci_move(part)
            elif part == 'startpos':
                self.initial_pos()
            elif part == 'fen':
                self.parse_fen(command.split('fen')[1].split('moves')[0])
            elif part == 'moves':
                is_move = True

    def evaluate_nnue(self):
        return nnue.nnue_evaluate_fen(bytes(self.positions[-1], encoding='utf-8'))

    def has_non_pawns(self):
        return any(p in self.board for p in [KNIGHT, BISHOP, ROOK, QUEEN])


#
# Search algorithm
#

class Searcher:
    def __init__(self):
        self.nodes = 0
        self.b_move = 0
        self.finish_time = 0
        self.stop_search = False

        self.history = {}  # History table for move ordering
        self.counter_hist = {}
        self.counter_move = {}
        self.killer = {}  # Killer move for each ply
        self.killer2 = {}
        self.tt = {}  # Transposition table
        self.threat = {}

        self.start_depth = 0

    def set_timer(self, ttime):
        self.finish_time = time.perf_counter() + ttime

    def refresh_timer(self, ttime):
        self.finish_time += ttime

    def move_values(self, game, movelist, ply, opp_move, hash_move):
        # Give every move a score for ordering
        killer_move = self.killer.get(ply)
        killer_move2 = self.killer2.get(ply)
        threat_move = self.threat.get(ply)
        counter = self.counter_move.get(opp_move)
        scores = {}
        for move in movelist:
            if move == hash_move:
                scores[move] = 2 * HIST_MAX + 5000
            elif move == threat_move:
                if game.is_capture(move):
                    scores[move] = 2 * HIST_MAX + 4990
                else:
                    scores[move] = 2 * HIST_MAX + 1
            elif game.is_capture(move):
                scores[move] = 2 * HIST_MAX + 2000 + piece_vals[game.board[move[1]] - 6] - piece_vals[
                    game.board[move[0]]]
            else:
                if move == killer_move:
                    scores[move] = 2 * HIST_MAX + 4
                elif move == killer_move2:
                    scores[move] = 2 * HIST_MAX + 3
                elif move == counter:
                    scores[move] = 2 * HIST_MAX + 2
                else:
                    scores[move] = self.history.get(move, 0)
                    scores[move] += self.counter_hist.get((opp_move, move), 0)

        return scores

    def qsearch(self, game, alpha, beta, ply):
        val = game.evaluate_nnue()

        self.nodes += 1

        if val >= beta:
            return val
        if alpha < val:
            alpha = val

        hash_move = None
        # Probe transposition table
        tt_entry = self.tt.get(game.positions[-1])
        if tt_entry:
            hash_move = tt_entry[1]
            if tt_entry[2] >= -ply:
                if tt_entry[3] == TT_EXACT or \
                        (tt_entry[3] == TT_LOWER and tt_entry[0] >= beta) or \
                        (tt_entry[3] == TT_UPPER and tt_entry[0] <= alpha):
                    return tt_entry[0]

        movelist = [m for m in game.movegen() if game.is_capture(m)]
        scores = self.move_values(game, movelist, 0, None, hash_move)
        movelist.sort(key=lambda x: scores[x], reverse=True)

        tt_flag = TT_UPPER
        best_move = None

        for move in movelist:

            cpy = game.copy()
            cpy.make_move(move)
            if cpy.in_check():
                continue
            cpy.rotate()
            cpy.positions.append(cpy.to_fen())
            self.nodes += 1

            score = -self.qsearch(cpy, -beta, -alpha, ply + 1)

            cpy.positions.pop()
            if score > val:
                val = score
                if score > alpha:
                    alpha = score
                    tt_flag = TT_EXACT
                    best_move = move
                    if score >= beta:
                        tt_flag = TT_LOWER
                        break
        if self.stop_search: return val
        self.tt[game.positions[-1]] = (val, best_move, -ply, tt_flag)

        return val

    def search(self, game, depth, alpha, beta, ply, do_pruning, opp_move, is_cut_node, root=False):

        repetitions = 0
        if not root:
            for pos in game.positions:
                if pos == game.positions[-1]:
                    repetitions += 1
                if repetitions > 1: return 0

        if depth <= 0:
            return self.qsearch(game, alpha, beta, ply)

        self.nodes += 1

        hash_move = None

        is_pv_node = beta - alpha != 1

        # Probe transposition table
        tt_entry = self.tt.get(game.positions[-1])
        if tt_entry:
            hash_move = tt_entry[1]
            if tt_entry[2] >= depth and not is_pv_node:
                if tt_entry[3] == TT_EXACT or \
                        (tt_entry[3] == TT_LOWER and tt_entry[0] >= beta) or \
                        (tt_entry[3] == TT_UPPER and tt_entry[0] <= alpha):
                    return tt_entry[0]

        if self.nodes % 10000 == 0:
            if len(self.tt) > TT_MAX:  # Clear transposition table if too many entries
                self.tt.clear()

        in_check = game.in_check()
        evalu = game.evaluate_nnue()

        self.threat[ply + 1] = None
        if not is_pv_node and do_pruning and not in_check:
            # Razoring
            if depth <= 3 and evalu + RAZOR_MARGIN < beta:
                score = self.qsearch(game, alpha, beta, ply)
                if score < beta:
                    return score

            # Reverse futility pruning
            if depth <= 6 and evalu >= beta + PRUNE_MARGIN * depth:
                return evalu

            # Null move pruning
            if depth >= 2 and evalu >= beta and game.has_non_pawns():
                cpy = game.copy()
                cpy.enp = 0
                cpy.rotate()
                reduction = 3 + depth // 3 + min(3, (evalu - beta) // PRUNE_MARGIN)
                cpy.positions.append(cpy.to_fen())
                score = -self.search(cpy, depth - reduction, -beta, -beta + 1, ply + 1, False, None, not is_cut_node)
                cpy.positions.pop()
                if score >= beta:
                    return score
                else:
                    tt_entry = self.tt.get(game.positions[-1])
                    if tt_entry:
                        self.threat[ply + 1] = tt_entry[1]

        best_move = None
        best_score = -MATE_SCORE
        legal_moves = 0
        tt_flag = TT_UPPER

        # Internal iterative reductions
        iir = not (root or hash_move or in_check) and (depth > 4 and is_cut_node or is_pv_node)

        # Generate and sort moves
        movelist = game.movegen()
        scores = self.move_values(game, movelist, ply, opp_move, hash_move)
        movelist.sort(key=lambda x: scores[x], reverse=True)

        for move in movelist:
            if self.nodes % 50000 == 0 and self.start_depth > 1 and time.perf_counter() > self.finish_time:
                self.stop_search = True
                break
            # Copy-Make
            cpy = game.copy()
            cpy.make_move(move)
            if cpy.in_check():
                continue
            legal_moves += 1
            cpy.rotate()

            check_move = cpy.in_check()

            prunable = not (is_pv_node or check_move or in_check or game.is_capture(
                move) or legal_moves == 1) and best_score > -MATE_SCORE + 100
            lmr_reduction = lmr(depth, legal_moves)
            if iir: lmr_reduction += 3 if is_pv_node else 1
            if is_pv_node: lmr_reduction -= 1

            # Futility pruning
            if prunable and depth - lmr_reduction <= 6 and evalu <= alpha - PRUNE_MARGIN - PRUNE_MARGIN * (
            max(0, depth - lmr_reduction)):
                continue

            # Counter move history pruning
            if prunable and depth - lmr_reduction <= 2 and self.counter_hist.get((opp_move, move), 0) <= 0 and scores[
                move] < -depth * depth:
                continue

            # Late move pruning
            if prunable and depth - lmr_reduction <= 1 and evalu <= alpha and legal_moves > 7:
                continue

            # Check extension
            ext = 1 if check_move and not root else 0

            cpy.positions.append(cpy.to_fen())

            # Principal Variation Search
            score = None
            if legal_moves == 1:
                score = -self.search(cpy, depth - 1 + ext, -beta, -alpha, ply + 1, True, move,
                                     False if is_pv_node else not is_cut_node)
            else:
                # Late Move Reductions for quiet moves
                reduction = 0
                if depth >= 3 and not in_check and not check_move and not game.is_capture(move):
                    reduction = lmr_reduction
                    if is_cut_node: reduction += 2
                    if root: reduction -= 1
                    reduction -= scores[move] // HIST_MAX
                    if reduction >= depth - 1 + ext:
                        reduction = depth - 2 + ext
                    reduction = max(0, reduction)

                score = -self.search(cpy, depth - 1 - reduction + ext, -alpha - 1, -alpha, ply + 1, True, move, True)
                if score > alpha and reduction != 0:
                    score = -self.search(cpy, depth - 1 + ext, -alpha - 1, -alpha, ply + 1, True, move, not is_cut_node)
                if score > alpha and score < beta:
                    score = -self.search(cpy, depth - 1 + ext, -beta, -alpha, ply + 1, True, move, False)
            cpy.positions.pop()

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    tt_flag = TT_EXACT
                    alpha = best_score
                    if score >= beta:
                        if not game.is_capture(move):  # Update history tables
                            self.history[move] = min(HIST_MAX, self.history.get(move, 0) + depth * depth)
                            if opp_move:
                                self.counter_move[opp_move] = move
                                self.counter_hist[(opp_move, move)] = min(HIST_MAX,
                                                                          self.counter_hist.get((opp_move, move),
                                                                                                0) + depth * depth)
                            k, k2 = self.killer.get(ply), self.killer2.get(ply)
                            if k != k2:
                                self.killer2[ply] = k
                                self.killer[ply] = move
                            else:
                                self.killer[ply] = move
                        tt_flag = TT_LOWER
                        for m in movelist:
                            if m == move: break
                            if game.is_capture(m): continue
                            self.history[m] = max(-HIST_MAX, self.history.get(m, 0) - depth * depth)
                            if opp_move:
                                self.counter_hist[(opp_move, m)] = max(-HIST_MAX, self.counter_hist.get((opp_move, m),
                                                                                                        0) - depth * depth)
                        break

        if self.stop_search: return best_score
        if legal_moves == 0:
            if game.in_check():
                return -MATE_SCORE + ply
            else:
                return 0

        if root:
            self.b_move = best_move

        # Update transposition table
        self.tt[game.positions[-1]] = (best_score, best_move, depth, tt_flag)

        return best_score

    def search_iterative(self, game, time_remaining):
        start_time = time.perf_counter()
        move_time = time_remaining / 30000
        tf_sum = 0
        if game.close_to_startpos(): move_time /= 2
        self.set_timer(move_time)

        # Reset history tables
        self.history = {}
        self.counter_hist = {}
        self.counter_move = {}
        self.killer = {}
        self.killer2 = {}
        self.threat = {}

        self.stop_search = False

        prev_move = None
        self.nodes = 0
        time_1 = time.perf_counter()

        alpha, beta = -MATE_SCORE, MATE_SCORE
        d = 1
        asp_cnt = 0
        prev_score = 0

        # Iterative Deepening with Aspiration Windows
        while d < 80:
            self.start_depth = d
            score = self.search(game, d, alpha, beta, 0, True, None, False, True)

            if time.perf_counter() > self.finish_time and d > 1:
                break

            time_2 = time.perf_counter()
            nps = int(self.nodes / (time_2 - time_1))
            time_elapsed = int((time.perf_counter() - start_time) * 1000)

            if d > 4 and not game.close_to_startpos():
                time_factor = 0.0
                if prev_move == self.b_move and asp_cnt == 0:
                    time_factor -= 0.2
                else:
                    time_factor += 0.1
                if prev_score > score:
                    time_factor += min(1.0, (prev_score - score) / PRUNE_MARGIN)
                if tf_sum > 1.5: time_factor = min(time_factor, 0.0)
                tf_sum += time_factor
                self.refresh_timer(move_time * time_factor)

            if score <= alpha:
                asp_cnt += 1
                alpha -= ASPIRATION_DELTA * 2 ** asp_cnt
                continue
            elif score >= beta:
                asp_cnt += 1
                beta += ASPIRATION_DELTA * 2 ** asp_cnt
                continue

            prev_move = self.b_move

            print('info depth {} time {} nodes {} nps {} score cp {} pv {}'.format(d, time_elapsed, self.nodes, nps,
                                                                                   int(score / 2.56),
                                                                                   game.uci_move(prev_move)))
            if d >= 4:
                alpha = score - ASPIRATION_DELTA
                beta = score + ASPIRATION_DELTA

            asp_cnt = 0
            d += 1
            prev_score = score

        print(f'bestmove {game.uci_move(prev_move)}')


def main():
    g = Game(None, True, True, True, True, True, 0, [])
    g.initial_pos()
    s = Searcher()

    while True:
        command = input()

        if command == 'uci':
            print('id name Habu')
            print('uciok')

        elif command == 'isready':
            print('readyok')

        elif command.startswith('go'):
            time_remaining = 30000
            c_split = command.split()
            for idx, val in enumerate(c_split):
                if val == 'wtime' and g.side:
                    time_remaining = float(c_split[idx + 1])
                elif val == 'btime' and not g.side:
                    time_remaining = float(c_split[idx + 1])
            s.search_iterative(g, time_remaining)

        elif command.startswith('position'):
            g.uci_position(command)

        elif command == 'quit':
            return


if __name__ == '__main__':
    main()


