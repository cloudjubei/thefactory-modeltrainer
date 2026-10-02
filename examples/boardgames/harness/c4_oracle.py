"""Exact values for Connect-4 measurements: the values already recorded in the label cache first, the native
solver for the rest, the deepest ply a certificate covers, and the solver's verdict on a disagreement between a net's
raw move and its own search."""
from __future__ import annotations

EMPTY_BOARD_VALUE = 1
EMPTY_BOARD_VALUE_SOURCE = "Connect-4 is a first-player win (Allis 1988; Allen 1988; Tromp's database)"
_BOOK = None


def solve(job: tuple) -> int:
    """Worker job: the exact value of (board, to_move) to the side to move, the project book short-circuiting what it
    proves."""
    global _BOOK
    from games.connect4 import C4State
    from harness import native_solver
    from harness.book import load_book

    if _BOOK is None:
        _BOOK = load_book("connect4")
    board, to_move = job
    return native_solver.solve_position(C4State(tuple(board), to_move, None, False), book=_BOOK)


def recorded_values(game, rows: list) -> dict:
    """{position key: exact value to the side to move} for every non-terminal position one move after a label-cache
    row; a row stores, per move, the value that move keeps for the mover, so its child is worth the negation."""
    from games.connect4 import C4State

    out = {}
    for row in rows:
        parent = C4State(tuple(row["board"]), row["to_move"], None, False)
        for a, v in row["values"].items():
            child = game.step(parent, int(a))
            if not game.is_terminal(child):
                out[game.state_key(child)] = -int(v)
    return out


def deepest_certified(cert: dict) -> int:
    """How many plies a certificate covers: its horizon when certified, else the ply of its first failure."""
    if cert["failures"]:
        return min(int(p) for p in cert["failures_by_ply"])
    if not cert["certified"]:
        raise ValueError("an incomplete walk with no failure certifies nothing")
    return cert["horizon"]


def move_optimal(game, state, move: int, value_of) -> bool:
    """Whether `move` keeps the position's exact value (`value_of`: a state → its value to the side to move)."""
    child = game.step(state, move)
    mover = game.current_player(state)
    kept = round(game.returns(child)[mover]) if game.is_terminal(child) else -value_of(child)
    return kept == value_of(state)


def classify(net_optimal: bool, search_optimal: bool) -> str:
    """The solver's verdict on a disagreement between the net's raw move and its search's preferred move."""
    return {(True, True): "both_optimal", (True, False): "search_wrong", (False, True): "net_wrong",
            (False, False): "both_wrong"}[(net_optimal, search_optimal)]
