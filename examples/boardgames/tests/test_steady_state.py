"""Direct tests for harness/steady_state.py — the priority-map rule language (win; else the single safe move; else
the first level holding exactly one safe move) and its solver-free verifier. Tic-tac-toe fixtures were checked by
hand: in X{0,1} O{2,3} with X to move, no move wins at once and every move is safe; only the centre (4) makes the fork
that wins, so the map {4: 0} wins and no other one-cell map does."""
from __future__ import annotations

import pytest

from games.connect4 import Connect4
from games.kalah import Kalah
from games.tictactoe import TicTacToe, TTTState
from harness.steady_state import Facts, choose, map_bits, simplify, verify


def _ttt(x, o, to_move=0):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=to_move, winner=None, done=False)


FORK = _ttt((0, 1), (2, 3))


@pytest.fixture
def facts():
    return Facts(TicTacToe())


def test_a_move_s_cell_is_the_one_board_cell_it_fills():
    ttt, c4 = Facts(TicTacToe()), Facts(Connect4())
    assert ttt.placed_cell(TicTacToe().initial_state(), 4) == 4
    root = Connect4().initial_state()
    assert c4.placed_cell(root, 3) == 3 and c4.placed_cell(Connect4().step(root, 3), 3) == 10


def test_a_game_whose_moves_do_not_fill_one_cell_is_refused():
    kalah = Kalah(3, 2)
    with pytest.raises(ValueError, match="placement"):
        Facts(kalah).placed_cell(kalah.initial_state(), 0)


def test_empty_cells_are_read_from_the_observation(facts):
    assert facts.empty_cells(FORK) == [4, 5, 6, 7, 8]


def test_a_winning_move_is_played_whatever_the_map(facts):
    s = _ttt((0, 1), (3, 4))
    assert choose(facts, s, {5: 0, 8: 0}, 2) == 2


def test_the_only_safe_move_is_played_whatever_the_map(facts):
    s = _ttt((0, 8), (4, 2))
    assert facts.of(s)[1] == [6] and choose(facts, s, {}, 2) == 6 and choose(facts, s, {3: 0}, 2) == 6


@pytest.mark.parametrize("levels,move", [({4: 0}, 4), ({4: 0, 5: 1}, 4), ({4: 0, 5: 0, 6: 1}, 6),
                                         ({4: 1, 5: 0}, 5), ({4: 0, 5: 0}, None), ({}, None), ({4: 2}, None)])
def test_the_first_level_holding_exactly_one_safe_move_decides(facts, levels, move):
    assert choose(facts, FORK, levels, 2) == move


def test_a_winning_map_is_verified_over_every_reply(facts):
    assert verify(facts, FORK, {4: 0}, 2) == {"won": True, "reason": None, "own_positions": 5, "positions": 10}


@pytest.mark.parametrize("levels,reason", [({}, "undefined"), ({5: 0}, "undefined"), ({6: 0}, "draw"),
                                           ({7: 0}, "undefined")])
def test_a_map_that_misses_a_line_is_rejected_with_the_first_failure(facts, levels, reason):
    r = verify(facts, FORK, levels, 2)
    assert not r["won"] and r["reason"] == reason


def test_a_forced_draw_is_rejected(facts):
    s = _ttt((0, 2, 3, 7), (1, 4, 5, 6))
    assert verify(facts, s, {}, 2)["reason"] == "draw"


def test_a_root_already_lost_is_rejected_as_a_loss(facts):
    lost = TTTState(board=(2, 2, 2, 1, 1, 0, 1, 0, 0), to_move=0, winner=1, done=True)
    assert verify(facts, lost, {}, 2)["reason"] == "loss"


def test_a_walk_past_the_cap_is_rejected(facts):
    assert verify(facts, FORK, {4: 0}, 2, cap=3)["reason"] == "cap"


def test_a_map_is_one_bit_per_empty_cell_plus_its_levels():
    assert map_bits({4: 0, 5: 1}, 5, 3) == 11 and map_bits({}, 30, 3) == 30
    with pytest.raises(ValueError, match="levelled"):
        map_bits({1: 0, 2: 0}, 1, 3)


def test_simplify_drops_every_level_the_win_does_not_need(facts):
    assert verify(facts, FORK, {4: 0, 5: 1, 6: 1}, 2)["won"]
    assert simplify(facts, FORK, {4: 0, 5: 1, 6: 1}, 2) == {4: 0}
    assert simplify(facts, FORK, {5: 0}, 2) == {5: 0}


def test_facts_are_computed_once_per_position(facts, monkeypatch):
    facts.of(FORK)
    monkeypatch.setattr(facts, "_wins_now", lambda *a: pytest.fail("recomputed"))
    assert facts.of(FORK)[1] == [4, 5, 6, 7, 8]
