# search.py
import time
import math
import constants as const


class Searcher:
    def __init__(self):
        self.nodes = 0
        self.b_move = 0
        self.finish_time = 0
        self.stop_search = False
        self.history = {}
        self.counter_hist = {}
        self.counter_move = {}
        self.killer = {}
        self.killer2 = {}
        self.tt = {}
        self.threat = {}
        self.start_depth = 0

    def set_timer(self, ttime):
        self.finish_time = time.perf_counter() + ttime

    def refresh_timer(self, ttime):
        self.finish_time += ttime

    def move_values(self, game, movelist, ply, opp_move, hash_move):
        killer_move = self.killer.get(ply)
        killer_move2 = self.killer2.get(ply)
        threat_move = self.threat.get(ply)
        counter = self.counter_move.get(opp_move)
        scores = {}
        for move in movelist:
            if move == hash_move:
                scores[move] = 2 * const.HIST_MAX + 5000
            elif move == threat_move:
                if game.is_capture(move):
                    scores[move] = 2 * const.HIST_MAX + 4990
                else:
                    scores[move] = 2 * const.HIST_MAX + 1
            elif game.is_capture(move):

                scores[move] = 2 * const.HIST_MAX + 2000 + const.piece_vals[game.board[move[1]] - 6] - \
                               const.piece_vals[game.board[move[0]]]
            else:
                if move == killer_move:
                    scores[move] = 2 * const.HIST_MAX + 4
                elif move == killer_move2:
                    scores[move] = 2 * const.HIST_MAX + 3
                elif move == counter:
                    scores[move] = 2 * const.HIST_MAX + 2
                else:
                    scores[move] = self.history.get(move, 0)
                    scores[move] += self.counter_hist.get((opp_move, move), 0)
        return scores

    def qsearch(self, game, alpha, beta, ply):
        val = game.evaluate_nnue()
        self.nodes += 1

        if val >= beta: return val
        if alpha < val: alpha = val

        hash_move = None
        tt_entry = self.tt.get(game.positions[-1])
        if tt_entry:
            hash_move = tt_entry[1]
            if tt_entry[2] >= -ply:  # depth TT >= -ply (depth qsearch coi như âm)
                if tt_entry[3] == const.TT_EXACT or \
                        (tt_entry[3] == const.TT_LOWER and tt_entry[0] >= beta) or \
                        (tt_entry[3] == const.TT_UPPER and tt_entry[0] <= alpha):
                    return tt_entry[0]

        movelist = [m for m in game.movegen() if game.is_capture(m)]
        scores = self.move_values(game, movelist, 0, None, hash_move)
        movelist.sort(key=lambda x: scores[x], reverse=True)

        tt_flag_q = const.TT_UPPER  # tt_flag cho qsearch
        best_move_q = None  # best_move cho qsearch

        for move in movelist:
            cpy = game.copy()
            cpy.make_move(move)
            if cpy.in_check(): continue

            cpy.rotate()
            cpy.positions.append(cpy.to_fen())
            score = -self.qsearch(cpy, -beta, -alpha, ply + 1)
            cpy.positions.pop()

            if score > val:
                val = score
                if score > alpha:
                    alpha = score
                    tt_flag_q = const.TT_EXACT
                    best_move_q = move
                    if score >= beta:
                        tt_flag_q = const.TT_LOWER
                        break

        if self.stop_search: return val
        self.tt[game.positions[-1]] = (val, best_move_q, -ply, tt_flag_q)
        return val

    def search(self, game, depth, alpha, beta, ply, do_pruning, opp_move, is_cut_node, root=False):
        repetitions = 0
        if not root:
            current_pos_fen = game.positions[-1]  # Tối ưu hóa
            for pos_fen_hist in game.positions:
                if pos_fen_hist == current_pos_fen:
                    repetitions += 1

            if repetitions > 1: return 0

        if depth <= 0:
            return self.qsearch(game, alpha, beta, ply)

        self.nodes += 1  # Đặt ở đây, mỗi vị trí search là 1 node.

        hash_move = None
        is_pv_node = beta - alpha != 1

        tt_entry = self.tt.get(game.positions[-1])
        if tt_entry:
            hash_move = tt_entry[1]
            if tt_entry[2] >= depth and not is_pv_node:
                if tt_entry[3] == const.TT_EXACT or \
                        (tt_entry[3] == const.TT_LOWER and tt_entry[0] >= beta) or \
                        (tt_entry[3] == const.TT_UPPER and tt_entry[0] <= alpha):
                    return tt_entry[0]

        if self.nodes % 10000 == 0:
            if len(self.tt) > const.TT_MAX:  # TT_MAX là 1e6
                self.tt.clear()

        in_check_current = game.in_check()
        evalu_static = game.evaluate_nnue()

        self.threat[ply + 1] = None  # Reset threat
        if not is_pv_node and do_pruning and not in_check_current:
            # Razoring
            if depth <= 3 and evalu_static + const.RAZOR_MARGIN < beta:
                score_q_razor = self.qsearch(game, alpha, beta, ply)
                if score_q_razor < beta:
                    return score_q_razor

            # Reverse futility pruning
            if depth <= 6 and evalu_static >= beta + const.PRUNE_MARGIN * depth:
                return evalu_static

            # Null move pruning
            if depth >= 2 and evalu_static >= beta and game.has_non_pawns():
                cpy_nmp = game.copy()
                cpy_nmp.enp = 0
                cpy_nmp.rotate()
                reduction_nmp = 3 + depth // 3 + min(3, (evalu_static - beta) // const.PRUNE_MARGIN)
                cpy_nmp.positions.append(cpy_nmp.to_fen())

                score_nmp_val = -self.search(cpy_nmp, depth - reduction_nmp, -beta, -beta + 1, ply + 1, False, None,
                                             not is_cut_node, False)
                cpy_nmp.positions.pop()
                if score_nmp_val >= beta:
                    return score_nmp_val
                else:
                    tt_entry_after_nmp = self.tt.get(game.positions[-1])  # Lấy TT của game hiện tại
                    if tt_entry_after_nmp:
                        self.threat[ply + 1] = tt_entry_after_nmp[1]

        best_move_node = None
        best_score_node = -const.MATE_SCORE
        legal_moves_played = 0
        tt_flag_node = const.TT_UPPER

        # IIR
        is_iir_candidate = not (root or hash_move or in_check_current)
        is_iir_depth_trigger = (depth > 4)
        is_iir_node_type_trigger = (is_cut_node or is_pv_node)
        iir_active = is_iir_candidate and is_iir_depth_trigger and is_iir_node_type_trigger

        movelist = game.movegen()
        scores_ordered = self.move_values(game, movelist, ply, opp_move, hash_move)
        movelist.sort(key=lambda x: scores_ordered[x], reverse=True)

        for move_idx, current_m in enumerate(movelist):
            if self.nodes % 50000 == 0 and self.start_depth > 1 and time.perf_counter() > self.finish_time:
                self.stop_search = True
                break

            cpy_iter = game.copy()
            cpy_iter.make_move(current_m)
            if cpy_iter.in_check(): continue

            legal_moves_played += 1
            cpy_iter.rotate()

            check_after_move = cpy_iter.in_check()

            # Prunable condition
            is_special_node = (is_pv_node or check_after_move or in_check_current or game.is_capture(
                current_m) or legal_moves_played == 1)
            is_score_high_enough = (best_score_node > -const.MATE_SCORE + 100)
            prunable_active = not is_special_node and is_score_high_enough

            # LMR reduction (dùng legal_moves_played)
            lmr_val = const.lmr(depth, legal_moves_played)
            if iir_active: lmr_val += 3 if is_pv_node else 1
            if is_pv_node: lmr_val -= 1

            # Futility
            if prunable_active and depth - lmr_val <= 6 and \
                    evalu_static <= alpha - const.PRUNE_MARGIN - const.PRUNE_MARGIN * (max(0, depth - lmr_val)):
                continue

            # Counter move history pruning
            if prunable_active and depth - lmr_val <= 2 and \
                    self.counter_hist.get((opp_move, current_m), 0) <= 0 and \
                    scores_ordered[current_m] < -(depth * depth):  # So sánh điểm sắp xếp
                continue

            # Late move pruning
            if prunable_active and depth - lmr_val <= 1 and evalu_static <= alpha and legal_moves_played > 7:
                continue

            ext = 1 if check_after_move and not root else 0

            cpy_iter.positions.append(cpy_iter.to_fen())

            # PVS
            search_score_val = 0
            if legal_moves_played == 1:  # Nước đầu tiên, full window

                child_is_cut_node = False if is_pv_node else not is_cut_node
                search_score_val = -self.search(cpy_iter, depth - 1 + ext, -beta, -alpha, ply + 1, True, current_m,
                                                child_is_cut_node, False)
            else:
                # LMR cho PVS
                reduction_pvs = 0
                if depth >= 3 and not in_check_current and not check_after_move and not game.is_capture(current_m):
                    reduction_pvs = lmr_val
                    if is_cut_node: reduction_pvs += 2
                    if root: reduction_pvs -= 1
                    reduction_pvs -= scores_ordered[current_m] // const.HIST_MAX

                    if reduction_pvs >= depth - 1 + ext:
                        reduction_pvs = depth - 2 + ext if depth - 2 + ext >= 0 else 0
                    reduction_pvs = max(0, reduction_pvs)

                search_score_val = -self.search(cpy_iter, depth - 1 - reduction_pvs + ext, -alpha - 1, -alpha, ply + 1,
                                                True, current_m, True, False)


                if search_score_val > alpha and reduction_pvs != 0:

                    search_score_val = -self.search(cpy_iter, depth - 1 + ext, -alpha - 1, -alpha, ply + 1, True,
                                                    current_m, not is_cut_node, False)

                if search_score_val > alpha and search_score_val < beta:  # Full window re-search

                    search_score_val = -self.search(cpy_iter, depth - 1 + ext, -beta, -alpha, ply + 1, True, current_m,
                                                    False, False)

            cpy_iter.positions.pop()

            if search_score_val > best_score_node:
                best_score_node = search_score_val
                best_move_node = current_m
                if best_score_node > alpha:
                    tt_flag_node = const.TT_EXACT
                    alpha = best_score_node
                    if best_score_node >= beta:  # Beta cutoff
                        tt_flag_node = const.TT_LOWER
                        if not game.is_capture(current_m):
                            self.history[current_m] = min(const.HIST_MAX,
                                                          self.history.get(current_m, 0) + depth * depth)
                            if opp_move:
                                self.counter_move[opp_move] = current_m
                                self.counter_hist[(opp_move, current_m)] = min(const.HIST_MAX,
                                                                               self.counter_hist.get(
                                                                                   (opp_move, current_m),
                                                                                   0) + depth * depth)


                            k_ply, k2_ply = self.killer.get(ply), self.killer2.get(ply)
                            if k_ply != k2_ply:
                                self.killer2[ply] = k_ply
                                self.killer[ply] = current_m
                            else:
                                self.killer[ply] = current_m

                        for m_penalty_idx in range(legal_moves_played - 1):
                            m_penalty = movelist[m_penalty_idx]
                            if m_penalty == current_m: break
                            for prev_tried_move_idx in range(move_idx):
                                prev_tried_m = movelist[prev_tried_move_idx]
                                if game.is_capture(prev_tried_m): continue
                                self.history[prev_tried_m] = max(-const.HIST_MAX,
                                                                 self.history.get(prev_tried_m, 0) - depth * depth)
                                if opp_move:
                                    self.counter_hist[(opp_move, prev_tried_m)] = max(-const.HIST_MAX,
                                                                                      self.counter_hist.get(
                                                                                          (opp_move, prev_tried_m),
                                                                                          0) - depth * depth)
                        break  # Beta cutoff

        if self.stop_search: return best_score_node
        if legal_moves_played == 0:
            return -const.MATE_SCORE + ply if in_check_current else 0

        if root:
            self.b_move = best_move_node

        self.tt[game.positions[-1]] = (best_score_node, best_move_node, depth, tt_flag_node)
        return best_score_node

    def search_iterative(self, game, time_remaining_ms):  # Đổi tên time_remaining
        start_time_id = time.perf_counter()  # Đổi tên start_time

        # Time management (giống file gốc)
        move_time_sec = time_remaining_ms / 30000.0  # Chuyển sang giây nếu time_remaining_ms là ms
        time_factor_sum = 0.0  # Đổi tên tf_sum
        if game.close_to_startpos(): move_time_sec /= 2.0
        self.set_timer(move_time_sec)  # set_timer nhận giây

        self.history.clear()
        self.counter_hist.clear()
        self.counter_move.clear()
        self.killer.clear()
        self.killer2.clear()
        self.threat.clear()
        self.stop_search = False

        best_move_from_prev_iter = None
        self.nodes = 0
        iter_start_time = time.perf_counter()

        alpha_id, beta_id = -const.MATE_SCORE, const.MATE_SCORE

        current_depth = 1
        aspiration_retry_count = 0
        score_from_prev_iter = 0

        while current_depth < 80:
            self.start_depth = current_depth

            current_alpha_win = alpha_id
            current_beta_win = beta_id
            if current_depth > 1 and aspiration_retry_count < 3:  # Giới hạn số lần thử cửa sổ hẹp
                current_alpha_win = score_from_prev_iter - const.ASPIRATION_DELTA * (2 ** aspiration_retry_count)
                current_beta_win = score_from_prev_iter + const.ASPIRATION_DELTA * (2 ** aspiration_retry_count)
                current_alpha_win = max(alpha_id, current_alpha_win)
                current_beta_win = min(beta_id, current_beta_win)

            search_val = self.search(game, current_depth, current_alpha_win, current_beta_win, 0, True, None, False,
                                     True)

            # Kiểm tra thời gian
            if time.perf_counter() > self.finish_time and current_depth > 1:
                break


            iter_end_time = time.perf_counter()
            if current_depth > 4 and not game.close_to_startpos():
                time_adj_factor = 0.0
                if best_move_from_prev_iter == self.b_move and aspiration_retry_count == 0:
                    time_adj_factor -= 0.2
                else:
                    time_adj_factor += 0.1
                if score_from_prev_iter > search_val:  # Điểm giảm
                    time_adj_factor += min(1.0, (score_from_prev_iter - search_val) / const.PRUNE_MARGIN)
                if time_factor_sum > 1.5: time_adj_factor = min(time_adj_factor, 0.0)
                time_factor_sum += time_adj_factor
                self.refresh_timer(move_time_sec * time_adj_factor)

            # Xử lý kết quả aspiration
            if search_val <= current_alpha_win and current_alpha_win > alpha_id:  # Fail low, và không phải đã ở biên alpha rộng nhất
                aspiration_retry_count += 1
                if search_val <= alpha_id:
                    aspiration_retry_count += 1
                    alpha_id = search_val - const.ASPIRATION_DELTA * (2 ** aspiration_retry_count)  # Mở rộng xuống
                    if current_depth > 0: continue  # Search lại cùng depth
                elif search_val >= beta_id:
                    aspiration_retry_count += 1
                    beta_id = search_val + const.ASPIRATION_DELTA * (2 ** aspiration_retry_count)  # Mở rộng lên
                    if current_depth > 0: continue  # Search lại cùng depth


            best_move_from_prev_iter = self.b_move  # Lưu best move của depth này

            time_elapsed_total_ms = int((time.perf_counter() - start_time_id) * 1000)
            nps_this_iter = 0
            if (iter_end_time - iter_start_time) > 0:
                nps_this_iter = int(self.nodes / (iter_end_time - iter_start_time))  # NPS cho lần lặp này

            pv_uci = game.uci_move(best_move_from_prev_iter) if best_move_from_prev_iter != 0 else "0000"
            score_cp_scaled = int(search_val / 2.56)

            print(
                f'info depth {current_depth} time {time_elapsed_total_ms} nodes {self.nodes} nps {nps_this_iter} score cp {score_cp_scaled} pv {pv_uci}')

            if current_depth >= 4:
                alpha_id = search_val - const.ASPIRATION_DELTA
                beta_id = search_val + const.ASPIRATION_DELTA

            aspiration_retry_count = 0  # Reset khi search thành công trong cửa sổ
            current_depth += 1
            score_from_prev_iter = search_val  # Lưu điểm cho lần sau
            iter_start_time = time.perf_counter()  # Reset thời điểm bắt đầu cho iter tiếp theo

        final_pv_uci = game.uci_move(best_move_from_prev_iter) if best_move_from_prev_iter != 0 else "0000"
        print(f'bestmove {final_pv_uci}')