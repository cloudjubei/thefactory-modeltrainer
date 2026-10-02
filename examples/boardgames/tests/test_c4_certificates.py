"""Direct tests for harness/c4_certificates.py — Allis-style value certificates for Connect-4 (track C): a written,
solver-free proof that, with White to move, Black can at least draw (claimeven, vertical and baseinverse instances
covering every group White could still complete), and the move flag built on it: a White move is NOT winning when
some Black reply reaches a certified position."""
from __future__ import annotations

import random

import pytest

from games.connect4 import COLS, ROWS, C4State, Connect4
from harness.c4_certificates import GROUPS, black_draw_certificate, column_units, move_not_winning

C4 = Connect4()


def _c4(columns: list, to_move: int) -> C4State:
    board = [0] * (ROWS * COLS)
    for c, stack in enumerate(columns):
        for r, ch in enumerate(stack):
            board[r * COLS + c] = {"x": 1, "o": 2}[ch]
    return C4State(tuple(board), to_move, None, False)


def _random_positions(n: int, stones: tuple, seed: int, to_move: int) -> list:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = C4.initial_state(rng)
        target = rng.randint(*stones)
        while not C4.is_terminal(s) and sum(1 for v in s.board if v) < target:
            s = C4.step(s, rng.choice(C4.legal_actions(s)))
        if not C4.is_terminal(s) and C4.current_player(s) == to_move:
            out.append(s)
    return out


def test_there_are_69_groups_of_four():
    assert len(GROUPS) == 69 and all(len(g) == 4 for g in GROUPS) and len(set(GROUPS)) == 69


@pytest.mark.parametrize("stack,units", [
    ("", [("claimeven", (0, 1)), ("claimeven", (2, 3)), ("claimeven", (4, 5))]),
    ("x", [("baseinverse", (1,)), ("claimeven", (2, 3)), ("claimeven", (4, 5))]),
    ("xo", [("claimeven", (2, 3)), ("claimeven", (4, 5))]),
    ("xox", [("baseinverse", (3,)), ("claimeven", (4, 5))]),
    ("xoxo", [("claimeven", (4, 5))]),
    ("xoxox", [("baseinverse", (5,))]),
    ("xoxoxo", []),
])
def test_a_column_s_empty_squares_split_into_its_forced_units(stack, units):
    assert column_units(_c4([stack, "", "", "", "", "", ""], 0), 0) == units


def test_the_empty_board_has_no_certificate_because_white_wins():
    assert not black_draw_certificate(C4.initial_state(random.Random(0)))


def test_a_certificate_is_only_for_white_to_move():
    with pytest.raises(ValueError, match="White to move"):
        black_draw_certificate(_c4(["x", "", "", "", "", "", ""], 1))


def test_a_position_with_only_even_columns_is_certified_by_claimevens_alone():
    cert = black_draw_certificate(_c4(["", "xo", "ox", "xooxoo", "oxooxo", "xx", "xx"], 0))
    assert cert and {kind for kind, _sq in cert} == {"claimeven"} and ("claimeven", (0, 7)) in cert


def test_the_loose_squares_are_matched_so_that_the_pair_inside_a_live_group_is_kept_together():
    cert = black_draw_certificate(_c4(["x", "xo", "ooxo", "xooxo", "xxxox", "oox", ""], 0))
    assert cert and sorted(sq for kind, sq in cert if kind == "baseinverse") == [(7, 39), (38, 26)]


@pytest.mark.parametrize("columns", [["xo", "xo", "xo", "", "", "", ""], ["ox", "ox", "ox", "x", "o", "o", "x"]])
def test_a_white_three_whose_last_square_black_cannot_claim_leaves_no_certificate(columns):
    assert black_draw_certificate(_c4(columns, 0)) is None


def test_every_certified_position_is_one_the_solver_does_not_score_as_a_white_win():
    from harness import native_solver

    certified = [s for s in _random_positions(300, (24, 40), seed=5, to_move=0) if black_draw_certificate(s)]
    assert len(certified) >= 5
    assert all(native_solver.solve_position(s) <= 0 for s in certified)


def test_a_flagged_move_is_never_a_winning_move_and_an_immediate_win_is_never_flagged():
    from harness import native_solver

    flagged = 0
    for s in _random_positions(40, (12, 26), seed=11, to_move=0):
        values = native_solver.move_values(s)
        for m in C4.legal_actions(s):
            if move_not_winning(C4, s, m):
                flagged += 1
                assert values[m] <= 0
    assert flagged >= 10
    win_now = _c4(["xxx", "ooo", "", "", "", "", "o"], 0)
    assert not move_not_winning(C4, win_now, 0)


def test_a_move_that_lets_black_win_at_once_is_flagged_and_the_move_that_wins_is_not():
    s = _c4(["x", "x", "x", "", "o", "o", "o"], 0)
    assert move_not_winning(C4, s, 0) and not move_not_winning(C4, s, 3)
