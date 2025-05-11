import constants as c
from nnue_interface import nnue_evaluate_fen


def invert_sides_on_board(board_array):
    """Helper function to invert pieces on a board array."""
    for i, piece in enumerate(board_array):
        if c.opp_piece(piece):
            board_array[i] -= 6
        elif c.my_piece(piece):
            board_array[i] += 6


class Game():
    def __init__(self, board=None, side=c.WHITE,
                 w_castle_k=True, w_castle_q=True,
                 b_castle_k=True, b_castle_q=True,
                 en_passant_sq=0, past_positions=None):
        self.board = board if board is not None else [0] * 120  # 10x12 board
        self.side = side  # True for White, False for Black

        self.w_castle_k = w_castle_k  # White King-side castle rights
        self.w_castle_q = w_castle_q  # White Queen-side castle rights
        self.b_castle_k = b_castle_k  # Black King-side castle rights
        self.b_castle_q = b_castle_q  # Black Queen-side castle rights

        self.enp = en_passant_sq  # En Passant square (0 if none)

        self.positions = past_positions if past_positions is not None else []  # For repetition checks & NNUE

    def copy(self):
        return Game(self.board.copy(), self.side,
                    self.w_castle_k, self.w_castle_q,
                    self.b_castle_k, self.b_castle_q,
                    self.enp, self.positions[:])  # Shallow copy of positions list

    def close_to_startpos(self):
        # Heuristic: count pieces in starting two ranks for both sides
        wcnt, bcnt = 0, 0
        # White's first two ranks (0-15 in 0-63)
        for i in range(16):
            if c.my_piece(self.board[c.mailbox64[i]]): wcnt += 1
        # Black's first two ranks (48-63 in 0-63, map to board)
        for i in range(48, 64):  # Black pieces are on 7th and 8th rank
            if c.opp_piece(self.board[c.mailbox64[i]]): bcnt += 1

        # If current side is black, the piece types are flipped
        if not self.side:
            wcnt, bcnt = bcnt, wcnt

        return wcnt >= 11 and bcnt >= 11  # e.g. more than 10 pieces on home ranks

    def initial_pos(self):
        self.parse_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    def uci_move(self, move_tuple):
        """Converts an internal move (from_sq, to_sq) to UCI string."""
        from_sq_10x12, to_sq_10x12 = move_tuple

        if self.side == c.WHITE:
            from_sq_0_63 = c.mailbox[from_sq_10x12]
            to_sq_0_63 = c.mailbox[to_sq_10x12]
        else:  # Black to move, board is already rotated for black's perspective
            from_sq_0_63 = c.mailbox[119 - from_sq_10x12]
            to_sq_0_63 = c.mailbox[119 - to_sq_10x12]

        move_str = c.square_name[from_sq_0_63] + c.square_name[to_sq_0_63]

        # Promotion (assuming queen promotion for simplicity as in original)
        piece_moved = self.board[from_sq_10x12]
        if piece_moved == c.PAWN and to_sq_10x12 in c.last_rank:
            move_str += 'q'  # Default to queen promotion
        return move_str

    def parse_fen(self, fen_string):
        self.board = [0] * 120  # Reset board, sentinel values for off-board
        for i in range(120):  # Fill with sentinel based on mailbox
            if c.mailbox[i] == -1:
                self.board[i] = -1  # Use -1 for off-board squares

        self.positions = []  # Reset history
        self.side = c.WHITE
        self.enp = 0
        self.w_castle_k = self.w_castle_q = self.b_castle_k = self.b_castle_q = False

        parts = fen_string.split()

        # Piece placement
        fen_board = parts[0]
        file = 0
        rank = 7  # FEN starts from 8th rank
        for char_piece in fen_board:
            if char_piece == '/':
                rank -= 1
                file = 0
            elif char_piece.isdigit():
                for _ in range(int(char_piece)):
                    sq64 = rank * 8 + file
                    self.board[c.mailbox64[sq64]] = c.EMPTY
                    file += 1
            else:
                sq64 = rank * 8 + file
                self.board[c.mailbox64[sq64]] = c.char_to_piece[char_piece]
                file += 1

        # Side to move
        if parts[1] == 'b':
            self.side = c.BLACK  # Will be rotated if black to move later

        # Castling rights
        castling_fen = parts[2]
        if 'K' in castling_fen: self.w_castle_k = True
        if 'Q' in castling_fen: self.w_castle_q = True
        if 'k' in castling_fen: self.b_castle_k = True
        if 'q' in castling_fen: self.b_castle_q = True

        # En passant
        if parts[3] != '-':
            sq_name = parts[3]
            col = ord(sq_name[0]) - ord('a')
            row = int(sq_name[1]) - 1
            sq64 = row * 8 + col
            # Map to 10x12. If black made the double push, EP target is on 3rd rank (from white's view)
            # If white made double push, EP target is on 6th rank (from white's view)
            self.enp = c.mailbox64[sq64]  # This will be correct after potential rotation

        # For FEN, if it's black's turn, we rotate the board to black's perspective.
        if self.side == c.BLACK:
            self.rotate()  # Rotate to current player's perspective

        self.positions.append(self.to_fen())  # Store initial FEN for repetition

    def to_fen(self):
        fen = ''
        # Create a temporary game state that is always from White's perspective for FEN generation
        temp_game = self.copy()
        if not self.side:  # If current side is Black, rotate to White's perspective
            temp_game.rotate()

        empty_count = 0
        for r in range(7, -1, -1):  # From 8th rank down to 1st
            for f in range(8):  # From a-file to h-file
                sq64 = r * 8 + f
                piece = temp_game.board[c.mailbox64[sq64]]

                if piece == c.EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen += c.fen_pieces[piece - 1]  # piece is 1-indexed

            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            if r > 0:
                fen += '/'

        fen += ' w ' if self.side else ' b '  # Use original side for this part

        castling_str = ""
        # Original castle rights, not temp_game's potentially flipped ones
        if self.w_castle_k: castling_str += 'K'
        if self.w_castle_q: castling_str += 'Q'
        if self.b_castle_k: castling_str += 'k'
        if self.b_castle_q: castling_str += 'q'
        fen += castling_str if castling_str else '-'

        fen += ' '
        if self.enp != 0:
            # enp square is from current player's perspective. Convert to absolute.
            enp_sq_0_63 = c.mailbox[self.enp if self.side else (119 - self.enp)]
            fen += c.square_name[enp_sq_0_63]
        else:
            fen += '-'

        # Halfmove clock and fullmove number are typically not used by NNUE for eval
        # but are part of standard FEN. For NNUE, often just pieces, side, castling, EP.
        fen += ' 0 1'  # Placeholder for halfmove clock and fullmove number
        return fen

    def rotate(self):
        """Rotates the board and flips side-dependent states."""
        invert_sides_on_board(self.board)
        self.board.reverse()  # Reverses the 10x12 list

        # Swap castling rights
        self.w_castle_k, self.w_castle_q, self.b_castle_k, self.b_castle_q = \
            self.b_castle_k, self.b_castle_q, self.w_castle_k, self.w_castle_q

        if self.enp != 0:
            self.enp = 119 - self.enp  # Flip en passant square

        self.side = not self.side

    def movegen(self):
        movelist = []
        for sq, piece_val in enumerate(self.board):
            if not c.my_piece(piece_val): continue

            piece_type = c.get_piece_type(piece_val)

            for d_offset in c.directions[piece_type]:
                next_sq = sq
                while True:
                    next_sq += d_offset

                    # Off board or own piece
                    if self.board[next_sq] == -1 or c.my_piece(self.board[next_sq]): break

                    if piece_type == c.PAWN:
                        # Standard pawn move (1 step forward)
                        if d_offset == c.N and self.board[next_sq] != c.EMPTY: break
                        # Double pawn push
                        if d_offset == c.N + c.N:
                            # Must be on 2nd rank (for current player) and path clear
                            # Original: sq < A2 or sq > H2 implies not on 2nd rank
                            # A2 is 81, H2 is 88 for white.
                            # For the current player, the "second rank" squares are 81-88.
                            is_on_second_rank = c.A2 <= sq <= c.H2
                            if not is_on_second_rank or \
                                    self.board[next_sq] != c.EMPTY or \
                                    self.board[next_sq - c.N] != c.EMPTY:  # Check intermediate square
                                break
                        # Pawn captures
                        if (d_offset == c.N + c.W or d_offset == c.N + c.E):
                            if not (c.opp_piece(self.board[next_sq]) or self.enp == next_sq):
                                break

                    movelist.append((sq, next_sq))

                    # If it's a capture or the piece is non-sliding
                    if c.opp_piece(self.board[next_sq]) or c.is_pawn_knight_king(piece_type):
                        break


        # Castling (King moves)
        # Assuming current player is White for rights checking, board is oriented for White
        king_sq_10x12 = -1
        for idx, p_val in enumerate(self.board):
            if p_val == c.KING:  # Current player's king
                king_sq_10x12 = idx
                break

        if king_sq_10x12 == c.E1:  # Standard king start square for current player
            # King-side castle (O-O)
            # Current player's rights: w_castle_k if white, b_castle_k if black (after rotate, it's w_castle_k)
            if self.w_castle_k and \
                    self.board[c.F1] == c.EMPTY and self.board[c.G1] == c.EMPTY and \
                    self.board[c.H1] == c.ROOK and \
                    not self.is_attacked(c.E1) and \
                    not self.is_attacked(c.F1) and \
                    not self.is_attacked(c.G1):
                movelist.append((c.E1, c.G1))  # King move e1g1

            # Queen-side castle (O-O-O)
            if self.w_castle_q and \
                    self.board[c.D1] == c.EMPTY and self.board[c.C1] == c.EMPTY and self.board[c.B1] == c.EMPTY and \
                    self.board[c.A1] == c.ROOK and \
                    not self.is_attacked(c.E1) and \
                    not self.is_attacked(c.D1) and \
                    not self.is_attacked(c.C1):
                movelist.append((c.E1, c.C1))  # King move e1c1
        return movelist

    def is_attacked(self, sq_10x12):
        """Checks if the given square (from current player's perspective) is attacked by opponent."""
        for piece_code_opp, dirs_opp in c.directions_isatt.items():  # piece_code_opp are black pieces 7-12
            for d_offset in dirs_opp:
                next_sq = sq_10x12
                while True:
                    next_sq += d_offset

                    board_val = self.board[next_sq]
                    if board_val == -1: break  # Off board
                    if c.my_piece(
                        board_val): break  # Blocked by own piece (from attacker's view, so current player's piece)

                    if board_val == piece_code_opp: return True  # Attacked by this piece

                    # Sliding piece checks
                    actual_attacker_type = c.get_piece_type(piece_code_opp)  # e.g. PAWN for PAWN+6

                    # Queen can attack as rook or bishop
                    if (actual_attacker_type == c.BISHOP or actual_attacker_type == c.ROOK) and \
                            board_val == c.QUEEN + 6:  # Opponent's queen
                        return True

                    if c.opp_piece(board_val): break  # Blocked by another opponent piece

                    # If attacker is P, N, or K, it doesn't slide further
                    if c.is_pawn_knight_king(actual_attacker_type): break
        return False

    def is_capture(self, move_tuple):
        _, to_sq = move_tuple
        return c.opp_piece(self.board[to_sq])

    def in_check(self):
        """Checks if the current player is in check."""
        king_sq = -1
        for i, piece_val in enumerate(self.board):
            if piece_val == c.KING:  # Current player's King
                king_sq = i
                break
        if king_sq == -1: return False  # Should not happen in a valid game
        return self.is_attacked(king_sq)

    def make_move(self, move_tuple):
        from_sq, to_sq = move_tuple

        moved_piece = self.board[from_sq]
        captured_piece = self.board[to_sq]  # Can be EMPTY

        # Update board
        self.board[to_sq] = moved_piece
        self.board[from_sq] = c.EMPTY

        # Handle en passant capture
        if moved_piece == c.PAWN and to_sq == self.enp and self.enp != 0:
            # The captured pawn is one square "behind" the EP target square
            # From current player's perspective, S is -10 if N is -10, so S is +10.
            # If N = -10, S = +10. Pawn captured at self.enp, actual pawn at self.enp + S
            self.board[to_sq + c.S] = c.EMPTY

            # Set new en passant square if pawn double push
        self.enp = 0  # Reset EP by default
        if moved_piece == c.PAWN and abs(to_sq - from_sq) == 2 * abs(c.N):  # Double push
            self.enp = from_sq + c.N  # EP square is one step behind destination

        # Pawn promotion
        if moved_piece == c.PAWN and to_sq in c.last_rank:
            self.board[to_sq] = c.QUEEN  # Auto-queen

        # Castling: move the rook
        if moved_piece == c.KING:
            if abs(to_sq - from_sq) == 2:  # King moved two squares, indicates castling
                if to_sq == c.G1:  # King-side castle (e1g1)
                    self.board[c.F1] = self.board[c.H1]
                    self.board[c.H1] = c.EMPTY
                elif to_sq == c.C1:  # Queen-side castle (e1c1)
                    self.board[c.D1] = self.board[c.A1]
                    self.board[c.A1] = c.EMPTY

        # Update castling rights (current player's perspective)
        if moved_piece == c.KING:
            self.w_castle_k = False
            self.w_castle_q = False

        if from_sq == c.H1 or to_sq == c.H1:  # H1 rook moved or captured
            self.w_castle_k = False
        if from_sq == c.A1 or to_sq == c.A1:  # A1 rook moved or captured
            self.w_castle_q = False
        if to_sq == c.A8: self.b_castle_q = False  # Black's queen-side rook captured
        if to_sq == c.H8: self.b_castle_k = False  # Black's king-side rook captured

        # Current position FEN is added after rotating for the next player
        # self.positions.append(self.to_fen()) # This was done in original make_uci_move, after rotate

    def make_uci_move(self, uci_move_str):
        """Parses a UCI move string and makes the move if legal."""
        # This requires parsing UCI (e.g., "e2e4", "e7e8q")
        generated_moves = self.movegen()
        target_move_tuple = None
        for m_tuple in generated_moves:
            if self.uci_move(m_tuple) == uci_move_str:
                target_move_tuple = m_tuple
                break

        if target_move_tuple:
            self.make_move(target_move_tuple)
            self.rotate()  # Switch side, rotate board
            self.positions.append(self.to_fen())  # Add FEN of new position
        else:
            print(f"Warning: Illegal UCI move {uci_move_str} for current position.")
            # Could print board and available moves for debugging
            # print(f"Current FEN: {self.to_fen()}")
            # print(f"Available moves: {[self.uci_move(m) for m in generated_moves]}")

    def uci_position(self, command_str):
        """Handles 'position' UCI command."""
        parts = command_str.split()

        fen_string = None
        moves_start_index = -1

        if "startpos" in parts:
            self.initial_pos()  # Sets up board and adds initial FEN to self.positions
            if "moves" in parts:
                moves_start_index = parts.index("moves") + 1
        elif "fen" in parts:
            fen_keyword_index = parts.index("fen")
            # FEN string can have up to 6 parts
            fen_parts = parts[fen_keyword_index + 1: fen_keyword_index + 7]
            fen_string = " ".join(fen_parts)
            self.parse_fen(fen_string)  # Sets up board and adds initial FEN to self.positions

            if "moves" in parts:
                moves_start_index = parts.index("moves") + 1

        if moves_start_index != -1:
            for i in range(moves_start_index, len(parts)):
                self.make_uci_move(parts[i])

    def evaluate_nnue(self):
        """Evaluates the current position using NNUE via the interface."""
        # The FEN for NNUE should be from the perspective of the side to move.
        # to_fen() already handles this.
        if not self.positions:  # Should not happen if parse_fen or initial_pos was called
            current_fen = self.to_fen()  # Generate if somehow missing
        else:
            current_fen = self.positions[-1]
        return nnue_evaluate_fen(current_fen)

    def has_non_pawns(self):
        """Checks if the current player has pieces other than pawns and king."""
        # This is used for null move pruning condition.
        # The board is from the current player's perspective.
        for piece_val in self.board:
            if c.my_piece(piece_val):
                ptype = c.get_piece_type(piece_val)
                if ptype not in [c.PAWN, c.KING, c.EMPTY, 0, -1]:  # 0 and -1 are empty/offboard
                    return True
        return False