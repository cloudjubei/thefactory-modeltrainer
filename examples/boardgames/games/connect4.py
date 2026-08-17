"""Connect 4 — the perfect-information, deterministic reference game.

Board is 6 rows x 7 columns, indexed `row * COLS + col` with row 0 the BOTTOM. An action is a column
0..6; a piece drops to the lowest empty cell. Cells hold 0 (empty), 1 (player 0), 2 (player 1).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

ROWS = 6
COLS = 7
_LINE_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))  # horizontal, vertical, two diagonals


@dataclass(frozen=True)
class C4State:
    board: tuple  # length ROWS*COLS, values in {0,1,2}
    to_move: int  # 0 or 1
    winner: int | None
    done: bool


class Connect4:
    name = "connect4"
    num_players = 2
    num_actions = COLS

    def initial_state(self, rng: random.Random | None = None) -> C4State:
        return C4State(board=(0,) * (ROWS * COLS), to_move=0, winner=None, done=False)

    def current_player(self, state: C4State) -> int:
        return state.to_move

    def legal_actions(self, state: C4State) -> list[int]:
        if state.done:
            return []
        top = (ROWS - 1) * COLS
        return [c for c in range(COLS) if state.board[top + c] == 0]

    def step(self, state: C4State, action: int, rng: random.Random | None = None) -> C4State:
        row = self._landing_row(state.board, action)
        if row < 0:
            raise ValueError(f"illegal move: column {action} is full or out of range")
        piece = state.to_move + 1
        board = list(state.board)
        board[row * COLS + action] = piece
        board = tuple(board)
        won = self._makes_line(board, row, action, piece)
        full = all(board[(ROWS - 1) * COLS + c] != 0 for c in range(COLS))
        return C4State(
            board=board,
            to_move=1 - state.to_move,
            winner=state.to_move if won else None,
            done=won or full,
        )

    def is_terminal(self, state: C4State) -> bool:
        return state.done

    def winner(self, state: C4State) -> int | None:
        return state.winner

    def returns(self, state: C4State) -> list[float]:
        if state.winner is None:
            return [0.0, 0.0]
        payoff = [-1.0, -1.0]
        payoff[state.winner] = 1.0
        return payoff

    def observation(self, state: C4State, player: int) -> list[float]:
        own = player + 1
        obs = [1.0 if v == own else -1.0 if v != 0 else 0.0 for v in state.board]
        obs.append(1.0 if state.to_move == player else 0.0)
        return obs

    def heuristic_action(self, state: C4State, rng: random.Random) -> int:
        """Win if a move wins now; else block the opponent's immediate win; else prefer the centre."""
        legal = self.legal_actions(state)
        me = state.to_move
        for col in legal:
            if self.step(state, col).winner == me:
                return col
        opp = 1 - me
        for col in legal:
            # Would the opponent win by playing col next? Simulate our move as a no-op passer: check whether the
            # opponent could complete a line at col — i.e. dropping the OPP piece there wins.
            row = self._landing_row(state.board, col)
            board = list(state.board)
            board[row * COLS + col] = opp + 1
            if self._makes_line(tuple(board), row, col, opp + 1):
                return col
        centre_order = sorted(legal, key=lambda c: abs(c - COLS // 2))
        best = centre_order[0]
        ties = [c for c in centre_order if abs(c - COLS // 2) == abs(best - COLS // 2)]
        return rng.choice(ties)

    def render(self, state: C4State) -> str:
        glyph = {0: ".", 1: "X", 2: "O"}
        rows = []
        for r in range(ROWS - 1, -1, -1):  # top row first
            rows.append(" ".join(glyph[state.board[r * COLS + c]] for c in range(COLS)))
        header = " ".join(str(c) for c in range(COLS))
        status = "draw" if state.done and state.winner is None else (
            f"{glyph[state.winner + 1]} wins" if state.done else f"{glyph[state.to_move + 1]} to move"
        )
        return "\n".join(rows) + "\n" + header + f"   [{status}]"

    def action_label(self, state: C4State, action: int) -> str:
        return f"col {action}"

    def state_key(self, state: C4State) -> tuple:
        """Canonical position key for the transposition table: the board + side to move (the board already
        determines whether the game is over)."""
        return (state.board, state.to_move)

    def symmetries(self) -> list[list[int]]:
        """The board's exploitable symmetries as ACTION (column) source-permutations `dest <- src`: identity and
        the left↔right mirror about the centre column. A game-theoretic value is invariant under these, so they
        both halve the search (canonical keys) and AUGMENT net training (a position + its mirror teach the same
        thing). A game with no symmetry returns just the identity."""
        return [list(range(COLS)), list(range(COLS - 1, -1, -1))]

    # --- SolvableGame hooks (the per-game parts the book engine needs; see docs/optimal-play-trainer-plan.md) ---
    def ply(self, state: C4State) -> int:
        """Stones placed — the frontier depth the book builder orders by (deepest, cheapest, first)."""
        return sum(1 for v in state.board if v != 0)

    def canonical_key(self, state: C4State) -> int:
        """The SYMMETRY-CANONICAL key (mirror-reduced), so the book stores a position and its reflection once."""
        from harness.solver import canonical_key, to_bitboard

        position, mask, _ = to_bitboard(state)
        return canonical_key(position, mask)

    def position_value(self, state: C4State, book=None) -> int:
        """Exact WEAK value to the player to move (win +1 / draw 0 / loss -1), via the bitboard solver, reading
        `book` for already-solved children (the bottom-up wall-break)."""
        from harness.solver import move_values

        vals = move_values(state, weak=True, book=book)
        return max(vals.values()) if vals else 0

    def exact_optimal_actions(self, state: C4State, max_empty: int) -> list[int] | None:
        """The game-theoretically OPTIMAL move set — but only when the position is cheap to solve exactly
        (≤ `max_empty` empty cells), else `None`. The opt-in seam a domain-oblivious search agent uses to play a
        PERFECT endgame (`MctsAgent(solve_endgame=…)`), so its residual deep-tactical misses can't lose an
        endgame it could have solved outright. `max_empty <= 0` disables it."""
        if state.done or max_empty <= 0:
            return None
        if sum(1 for v in state.board if v == 0) > max_empty:
            return None
        from harness.solver import optimal_columns

        return optimal_columns(state) or None

    # --- internals ---
    def _landing_row(self, board: tuple, col: int) -> int:
        if not 0 <= col < COLS:
            return -1
        for r in range(ROWS):
            if board[r * COLS + col] == 0:
                return r
        return -1

    def _makes_line(self, board: tuple, row: int, col: int, piece: int) -> bool:
        for dr, dc in _LINE_DIRS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < ROWS and 0 <= c < COLS and board[r * COLS + c] == piece:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= 4:
                return True
        return False
