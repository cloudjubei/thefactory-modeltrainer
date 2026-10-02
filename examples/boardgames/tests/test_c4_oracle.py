"""Direct tests for harness/c4_oracle.py — the exact-value plumbing the Connect-4 measurements share: values already
recorded in the label cache, the solver for the rest, the deepest ply a certificate covers, and the solver's verdict
on a disagreement between a net and its search."""
from __future__ import annotations

import random

import pytest

from games.connect4 import COLS, ROWS, C4State, Connect4
from games.tictactoe import TicTacToe, TTTState
from harness.c4_oracle import classify, deepest_certified, move_optimal, recorded_values, solve

C4 = Connect4()
TTT = TicTacToe()


def _c4(columns: list, to_move: int) -> C4State:
    board = [0] * (ROWS * COLS)
    for c, stack in enumerate(columns):
        for r, ch in enumerate(stack):
            board[r * COLS + c] = {"x": 1, "o": 2}[ch]
    return C4State(tuple(board), to_move, None, False)


def test_recorded_values_give_each_child_the_negated_value_its_move_keeps_for_the_mover():
    s = _c4(["xx", "oo", "", "", "", "", ""], 0)
    rows = [{"board": list(s.board), "to_move": 0, "values": {"0": 1, "3": 0, "6": -1}}]
    got = recorded_values(C4, rows)
    for a, v in ((0, 1), (3, 0), (6, -1)):
        assert got[C4.state_key(C4.step(s, a))] == -v
    assert len(got) == 3


def test_a_move_that_ends_the_game_is_not_a_recorded_child():
    s = _c4(["xxx", "ooo", "", "", "", "", ""], 0)
    rows = [{"board": list(s.board), "to_move": 0, "values": {"0": 1, "1": 0}}]
    assert list(recorded_values(C4, rows)) == [C4.state_key(C4.step(s, 1))]


def test_the_solver_job_returns_the_exact_value_to_the_side_to_move():
    from harness import native_solver

    s = C4.initial_state(random.Random(0))
    for a in (3, 3, 2, 4, 2, 4, 1, 5, 1, 5, 0, 6, 0, 6, 3, 3, 2, 2, 4, 4):
        s = C4.step(s, a)
    assert solve((s.board, s.to_move)) == native_solver.solve_position(s)


@pytest.mark.parametrize("cert,deepest", [
    ({"certified": True, "horizon": 10, "failures": 0, "failures_by_ply": {}}, 10),
    ({"certified": False, "horizon": 10, "failures": 3, "failures_by_ply": {4: 1, 6: 2}}, 4),
    ({"certified": False, "horizon": 10, "failures": 1, "failures_by_ply": {"0": 1}}, 0),
])
def test_the_deepest_certified_ply_is_where_the_first_failure_sits(cert, deepest):
    assert deepest_certified(cert) == deepest


def test_a_walk_cut_short_is_never_counted_as_certified_to_its_horizon():
    with pytest.raises(ValueError, match="incomplete"):
        deepest_certified({"certified": False, "horizon": 10, "failures": 0, "failures_by_ply": {}})


@pytest.mark.parametrize("net,search,label", [(True, True, "both_optimal"), (True, False, "search_wrong"),
                                              (False, True, "net_wrong"), (False, False, "both_wrong")])
def test_a_disagreement_is_classified_by_which_side_misses_the_optimum(net, search, label):
    assert classify(net, search) == label


def test_a_move_is_optimal_exactly_when_it_keeps_the_position_s_value():
    s = TTTState((1, 1, 0, 2, 2, 0, 0, 0, 0), 0, None, False)
    value = lambda st: TTT.position_value(st)
    assert move_optimal(TTT, s, 2, value)
    assert not move_optimal(TTT, s, 5, value)
    fork = TTTState((1, 2, 0, 0, 2, 0, 0, 1, 0), 0, None, False)
    assert value(fork) == 1 and move_optimal(TTT, fork, 6, value) and not move_optimal(TTT, fork, 2, value)
    drawn = TTTState((1, 0, 0, 0, 2, 0, 0, 0, 0), 0, None, False)
    assert move_optimal(TTT, drawn, 8, value) and move_optimal(TTT, drawn, 2, value)
