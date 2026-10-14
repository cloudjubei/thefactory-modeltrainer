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


@pytest.mark.parametrize("levels,move", [({4: 0}, 4), ({4: 0, 5: 0}, 4), ({5: 0, 4: 0, 6: 1}, 4),
                                         ({4: 1, 5: 1, 8: 0}, 8), ({7: 1, 5: 1}, 5), ({}, None), ({4: 2}, None)])
def test_with_ties_to_the_lowest_the_first_level_holding_any_safe_move_decides(levels, move):
    assert choose(Facts(TicTacToe(), ties="lowest"), FORK, levels, 2) == move


def test_the_rule_reads_exactly_one_safe_move_a_level_unless_told_otherwise(facts):
    assert facts.ties == "exact" and choose(facts, FORK, {4: 0, 5: 0}, 2) is None
    with pytest.raises(ValueError, match="ties"):
        Facts(TicTacToe(), ties="highest")


def test_ties_change_only_the_levels_not_wins_or_the_one_safe_move():
    lowest = Facts(TicTacToe(), ties="lowest")
    assert choose(lowest, _ttt((0, 1), (3, 4)), {5: 0, 8: 0}, 2) == 2
    assert choose(lowest, _ttt((0, 8), (4, 2)), {3: 0, 7: 0}, 2) == 6


def test_with_ties_to_the_lowest_a_map_verifies_that_the_exact_rule_leaves_undefined():
    assert verify(Facts(TicTacToe()), FORK, {4: 0, 5: 0}, 2)["reason"] == "undefined"
    assert verify(Facts(TicTacToe(), ties="lowest"), FORK, {4: 0, 5: 0}, 2)["won"]


DEEP = Facts(TicTacToe(), safe_depth=2)


def test_safety_is_one_ply_unless_told_otherwise(facts):
    corners = _ttt((0, 8), (4,), to_move=1)
    assert facts.safe_depth == 1 and facts.of(corners)[1] == [1, 2, 3, 5, 6, 7]
    with pytest.raises(ValueError, match="safe_depth"):
        Facts(TicTacToe(), safe_depth=3)


def test_two_ply_safety_drops_a_move_the_opponent_answers_with_a_win_no_reply_stops():
    """X on opposite corners, O in the centre: a corner lets X fork (two threats, one block), an edge does not."""
    corners = _ttt((0, 8), (4,), to_move=1)
    assert DEEP.of(corners)[1] == [1, 3, 5, 7] and DEEP.of(corners)[2] == {1: 1, 3: 3, 5: 5, 7: 7}
    assert DEEP.of(_ttt((0, 2), (1,), to_move=1))[1] == [4]


def test_two_ply_safety_keeps_the_one_ply_safe_moves_when_every_one_of_them_loses_in_two():
    lost = _ttt((0, 5, 7), (1, 2), to_move=1)
    assert DEEP.of(lost)[1] == Facts(TicTacToe()).of(lost)[1] == [3, 4, 6, 8]


def test_two_ply_safety_changes_neither_the_wins_nor_a_forced_safe_move():
    assert DEEP.of(_ttt((0, 1), (3, 4)))[0] == [2]
    assert DEEP.of(_ttt((0, 8), (4, 2)))[1] == [6]


def test_an_answer_that_fills_the_board_is_a_draw_not_a_threat():
    """O to move with two empties: 6 wins at once; after 3, X's only answer, 6, fills the board — a draw, so 3 stays
    safe beside 6."""
    assert DEEP.of(_ttt((0, 1, 4, 5), (2, 7, 8), to_move=1))[:2] == ([6], [3, 6])


def test_with_two_ply_safety_the_only_safe_move_is_played_whatever_the_map():
    s = _ttt((0, 2), (1,), to_move=1)
    assert choose(DEEP, s, {3: 0, 5: 0}, 2) == 4 and choose(Facts(TicTacToe()), s, {3: 0}, 2) == 3


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


def test_the_facts_cache_is_emptied_before_it_outgrows_its_limit_and_answers_stay_right():
    game = TicTacToe()
    small, plain = Facts(game, limit=2), Facts(game, limit=1_000)
    states = [_ttt((i,), ()) for i in range(1, 5)]
    for s in states:
        assert small.of(s) == plain.of(s)
        assert len(small._facts) <= 2
    assert small.cleared >= 1
    assert all(small.of(s) == plain.of(s) for s in states)


def test_the_facts_cache_keeps_everything_below_its_limit():
    game = TicTacToe()
    facts = Facts(game, limit=4)
    for i in range(1, 5):
        facts.of(_ttt((i,), ()))
    assert len(facts._facts) == 4 and facts.cleared == 0


def test_the_facts_cache_is_bounded_by_default():
    assert Facts(TicTacToe()).limit == 2_000_000


def test_a_verify_past_its_deadline_is_a_timeout(facts):
    import time

    r = verify(facts, FORK, {4: 0}, 2, deadline=time.monotonic() - 1)
    assert r["won"] is False and r["reason"] == "timeout"


def test_a_verify_within_its_deadline_is_unchanged(facts):
    import time

    assert verify(facts, FORK, {4: 0}, 2, deadline=time.monotonic() + 60)["won"]


def test_simplify_past_its_deadline_keeps_every_level(facts):
    import time

    levels = {4: 0, 5: 1, 6: 1}
    assert simplify(facts, FORK, levels, 2, deadline=time.monotonic() - 1) == levels
    assert simplify(facts, FORK, levels, 2) == {4: 0}


def test_simplify_uses_the_given_verifier(facts):
    calls = []

    def verifier(*a, **k):
        calls.append(a[2])
        return verify(*a, **k)

    assert simplify(facts, FORK, {4: 0, 5: 1}, 2, verifier=verifier) == {4: 0} and calls
