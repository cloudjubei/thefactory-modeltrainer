"""Allis-style VALUE CERTIFICATES for Connect-4 (track C): a written, solver-free proof that Black, the side that
controls zugzwang when White is to move, can at least draw.

A certificate is a pairing strategy over every empty square: whenever White plays one square of a pair, Black plays
the other at once. Black then owns at least one square of every pair, and the upper square of every same-column
pair. A pair is only usable if its partner is always playable the moment White takes the first square, which forces
each column's partition:
  claimeven    (r, r+1), r even (0-based): White must fill r first, Black takes r+1;
  baseinverse  two directly playable squares in different columns: Black takes whichever White leaves.
A column of even height splits into claimevens; one of odd height leaves its bottom square loose, and the loose
squares (always an even number with White to move) are matched into baseinverses. Allis's vertical (an odd-even
pair) cannot appear: it would leave a top square no reply could answer.

The certificate holds when every group White could still complete holds a Black stone, a claimeven's upper square,
or both squares of one baseinverse."""
from __future__ import annotations

from functools import lru_cache

from games.connect4 import COLS, ROWS

BLACK = 2


def _groups() -> tuple:
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    out.append(tuple(rr * COLS + cc for rr, cc in cells))
    return tuple(out)


GROUPS = _groups()


def _height(board: tuple, c: int) -> int:
    return sum(1 for r in range(ROWS) if board[r * COLS + c])


def column_units(state, c: int) -> list:
    """Column c's empty squares as the units the pairing forces, by row: ("claimeven", (r, r+1)) pairs and, at the
    bottom of a column of odd height, one ("baseinverse", (r,)) square left for a cross-column pair."""
    return _units(state.board, c)


def _units(board: tuple, c: int) -> list:
    h = _height(board, c)
    units = []
    if h % 2:
        units.append(("baseinverse", (h,)))
        h += 1
    units += [("claimeven", (r, r + 1)) for r in range(h, ROWS, 2)]
    return units


def _matchings(items: list):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for i, other in enumerate(rest):
        for tail in _matchings(rest[:i] + rest[i + 1:]):
            yield [(first, other)] + tail


@lru_cache(maxsize=1 << 16)
def _certificate(board: tuple):
    claimed, loose, claimevens = set(), [], []
    for c in range(COLS):
        for kind, rows in _units(board, c):
            if kind == "claimeven":
                lower, upper = (r * COLS + c for r in rows)
                claimed.add(upper)
                claimevens.append(("claimeven", (lower, upper)))
            else:
                loose.append(rows[0] * COLS + c)
    live = [g for g in GROUPS if not any(board[s] == BLACK or s in claimed for s in g)]
    for matching in _matchings(loose):
        if all(any(a in g and b in g for a, b in matching) for g in live):
            return tuple(claimevens + [("baseinverse", pair) for pair in matching])
    return None


def black_draw_certificate(state):
    """The rule instances proving Black at least draws from this White-to-move position, or None if the pairing
    proves nothing (which says nothing about the value)."""
    if state.to_move != 0:
        raise ValueError("a Black certificate is for positions with White to move")
    return _certificate(state.board)


def move_not_winning(game, state, move: int) -> bool:
    """True when White's `move` is proven not to win: some Black reply ends the game without a White win or reaches
    a certified position."""
    child = game.step(state, move)
    if game.is_terminal(child):
        return game.winner(child) != state.to_move
    for reply in game.legal_actions(child):
        after = game.step(child, reply)
        if game.is_terminal(after) or black_draw_certificate(after):
            return True
    return False
