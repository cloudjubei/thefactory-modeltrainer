"""Tic-tac-toe — the smallest SOLVABLE game, the second game that proves the optimal-play engine generalizes.

3x3 board indexed `row * 3 + col`; an action is a cell 0..8. Cells hold 0 (empty), 1 (player 0), 2 (player 1).
Unlike Connect 4 its ENTIRE game tree solves in milliseconds, so its book completes and a provably-optimal
book-agent falls straight out — the clean, complete demonstration of the same tablebase + symmetry + early-
termination machinery. It exercises the full 8-fold DIHEDRAL symmetry (4 rotations × 2 reflections), a richer
group than Connect 4's single mirror, via the same `canonical_key` / `symmetries` SolvableGame hooks.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

N = 3
CELLS = N * N
_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


def _d4_perms() -> list[list[int]]:
    """The 8 dihedral symmetries of the 3x3 grid as cell permutations `perm[src] = dest` (forward maps). As a
    group they're closed under inverse, so the same set augments a net and canonicalises the book."""
    maps = [
        lambda r, c: (r, c),          # identity
        lambda r, c: (c, N - 1 - r),  # rotate 90
        lambda r, c: (N - 1 - r, N - 1 - c),  # rotate 180
        lambda r, c: (N - 1 - c, r),  # rotate 270
        lambda r, c: (r, N - 1 - c),  # mirror columns
        lambda r, c: (N - 1 - r, c),  # mirror rows
        lambda r, c: (c, r),          # transpose
        lambda r, c: (N - 1 - c, N - 1 - r),  # anti-transpose
    ]
    perms = []
    for m in maps:
        p = [0] * CELLS
        for r in range(N):
            for c in range(N):
                nr, nc = m(r, c)
                p[r * N + c] = nr * N + nc
        perms.append(p)
    return perms


_D4 = _d4_perms()


def _transform(board: tuple, perm: list[int]) -> tuple:
    t = [0] * CELLS
    for s in range(CELLS):
        t[perm[s]] = board[s]
    return tuple(t)


def _board_int(board: tuple) -> int:
    v = 0
    for cell in board:
        v = v * 3 + cell
    return v


@dataclass(frozen=True)
class TTTState:
    board: tuple  # length 9, values in {0,1,2}
    to_move: int  # 0 or 1
    winner: int | None
    done: bool


class TicTacToe:
    name = "tictactoe"
    num_players = 2
    num_actions = CELLS

    def initial_state(self, rng: random.Random | None = None) -> TTTState:
        return TTTState(board=(0,) * CELLS, to_move=0, winner=None, done=False)

    def current_player(self, state: TTTState) -> int:
        return state.to_move

    def legal_actions(self, state: TTTState) -> list[int]:
        if state.done:
            return []
        return [i for i in range(CELLS) if state.board[i] == 0]

    def step(self, state: TTTState, action: int, rng: random.Random | None = None) -> TTTState:
        if not (0 <= action < CELLS) or state.board[action] != 0:
            raise ValueError(f"illegal move: cell {action} is taken or out of range")
        piece = state.to_move + 1
        board = list(state.board)
        board[action] = piece
        board = tuple(board)
        won = self._wins(board, piece)
        full = all(v != 0 for v in board)
        return TTTState(board=board, to_move=1 - state.to_move, winner=state.to_move if won else None, done=won or full)

    def is_terminal(self, state: TTTState) -> bool:
        return state.done

    def winner(self, state: TTTState) -> int | None:
        return state.winner

    def returns(self, state: TTTState) -> list[float]:
        if state.winner is None:
            return [0.0, 0.0]
        payoff = [-1.0, -1.0]
        payoff[state.winner] = 1.0
        return payoff

    def observation(self, state: TTTState, player: int) -> list[float]:
        own = player + 1
        obs = [1.0 if v == own else -1.0 if v != 0 else 0.0 for v in state.board]
        obs.append(1.0 if state.to_move == player else 0.0)
        return obs

    def heuristic_action(self, state: TTTState, rng: random.Random) -> int:
        legal = self.legal_actions(state)
        me = state.to_move
        for a in legal:
            if self.step(state, a).winner == me:
                return a
        opp = 1 - me
        for a in legal:  # block an immediate opponent win
            board = list(state.board)
            board[a] = opp + 1
            if self._wins(tuple(board), opp + 1):
                return a
        for pref in (4, 0, 2, 6, 8):  # centre, then corners
            if pref in legal:
                return pref
        return rng.choice(legal)

    def render(self, state: TTTState) -> str:
        glyph = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(glyph[state.board[r * N + c]] for c in range(N)) for r in range(N)]
        status = "draw" if state.done and state.winner is None else (
            f"{glyph[state.winner + 1]} wins" if state.done else f"{glyph[state.to_move + 1]} to move"
        )
        return "\n".join(rows) + f"   [{status}]"

    def action_label(self, state: TTTState, action: int) -> str:
        return f"cell {action}"

    def state_key(self, state: TTTState) -> tuple:
        return (state.board, state.to_move)

    # --- SolvableGame hooks -------------------------------------------------------------------------------
    def ply(self, state: TTTState) -> int:
        return sum(1 for v in state.board if v != 0)

    def symmetries(self) -> list[list[int]]:
        return _D4

    def canonical_key(self, state: TTTState) -> int:
        """The DIHEDRAL-canonical key: the smallest base-3 board encoding over all 8 symmetries, plus side to
        move. A position and its 7 rotations/reflections collapse to one entry."""
        best = min(_board_int(_transform(state.board, p)) for p in _D4)
        return best * 2 + state.to_move

    def position_value(self, state: TTTState, book=None) -> int:
        """Exact value to the player to move (win +1 / draw 0 / loss -1) — a memoised minimax over the tiny tree.
        `book` is accepted for interface parity; the tree is small enough not to need it."""
        return self._minimax(state, {})

    def exact_optimal_actions(self, state: TTTState, max_empty: int) -> list[int] | None:
        if state.done or max_empty <= 0:
            return None
        if sum(1 for v in state.board if v == 0) > max_empty:
            return None
        return self._optimal_actions(state) or None

    # --- internals ---
    def _wins(self, board: tuple, piece: int) -> bool:
        return any(board[a] == piece and board[b] == piece and board[c] == piece for a, b, c in _LINES)

    def _child_value(self, state: TTTState, action: int, memo: dict) -> int:
        me = state.to_move
        child = self.step(state, action)
        if child.done:
            return 1 if child.winner == me else 0  # I win now, or fill to a draw (I can't lose on my own move)
        return -self._minimax(child, memo)

    def _minimax(self, state: TTTState, memo: dict) -> int:
        if state.done:
            me = state.to_move
            return 0 if state.winner is None else (1 if state.winner == me else -1)
        key = self.canonical_key(state)
        cached = memo.get(key)
        if cached is not None:
            return cached
        best = -1
        for a in self.legal_actions(state):
            v = self._child_value(state, a, memo)
            if v > best:
                best = v
            if best == 1:
                break
        memo[key] = best
        return best

    def _optimal_actions(self, state: TTTState) -> list[int]:
        memo: dict = {}
        vals = {a: self._child_value(state, a, memo) for a in self.legal_actions(state)}
        best = max(vals.values())
        return sorted(a for a, v in vals.items() if v == best)
