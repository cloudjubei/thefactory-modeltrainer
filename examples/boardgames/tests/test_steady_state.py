"""Direct tests for harness/steady_state.py — the priority-map rule language (win; else the single safe move; else
the first level holding exactly one safe move) and its solver-free verifier. Tic-tac-toe fixtures were checked by
hand: in X{0,1} O{2,3} with X to move, no move wins at once and every move is safe; only the centre (4) makes the fork
that wins, so the map {4: 0} wins and no other one-cell map does."""
from __future__ import annotations

import itertools

import pytest

from games.connect4 import Connect4
from games.kalah import Kalah
from games.tictactoe import TicTacToe, TTTState
from harness.steady_state import SWITCH, Facts, choose, map_bits, phase_cell, restrict, simplify, verify


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


LAST = Facts(TicTacToe(), ties="lowest", unlevelled="last")


def test_unlevelled_cells_are_passed_over_unless_told_otherwise(facts):
    assert facts.unlevelled == "none" and choose(Facts(TicTacToe(), ties="lowest"), FORK, {}, 2) is None
    with pytest.raises(ValueError, match="unlevelled"):
        Facts(TicTacToe(), unlevelled="first")


@pytest.mark.parametrize("levels,move", [({}, 4), ({5: 0}, 5), ({5: 1, 7: 1}, 5), ({4: 0, 5: 0, 6: 1}, 4),
                                         ({4: 1, 5: 1}, 4), ({4: 2, 5: 2, 6: 2, 7: 2, 8: 2}, None)])
def test_with_unlevelled_cells_last_they_form_a_level_after_the_map_s(levels, move):
    assert choose(LAST, FORK, levels, 2) == move


def test_the_last_level_is_read_like_any_other():
    exact = Facts(TicTacToe(), unlevelled="last")
    assert choose(exact, FORK, {}, 2) is None and choose(exact, FORK, {4: 0, 5: 0, 6: 0, 7: 0}, 2) == 8
    assert choose(exact, FORK, {4: 0, 5: 0, 6: 0}, 2) is None


def test_unlevelled_cells_last_change_neither_the_wins_nor_a_forced_safe_move():
    assert choose(LAST, _ttt((0, 1), (3, 4)), {5: 0}, 2) == 2
    assert choose(LAST, _ttt((0, 8), (4, 2)), {3: 0}, 2) == 6


def test_with_unlevelled_cells_last_the_empty_map_verifies_where_it_was_undefined():
    assert verify(LAST, FORK, {}, 2)["won"]
    assert verify(Facts(TicTacToe(), ties="lowest"), FORK, {}, 2)["reason"] == "undefined"


FORCING = Facts(TicTacToe(), forcing=True)


def test_every_safe_move_is_kept_unless_told_to_prefer_forcing_ones(facts):
    assert facts.forcing is False and facts.of(FORK)[1] == [4, 5, 6, 7, 8]
    with pytest.raises(ValueError, match="forcing"):
        Facts(TicTacToe(), forcing="yes")


def test_preferring_forcing_moves_keeps_the_safe_moves_that_leave_the_opponent_at_most_one_safe_reply():
    """In FORK the centre makes two threats (no safe reply for O), 7 and 8 one each (one safe reply); 5 and 6 none."""
    assert FORCING.of(FORK)[1] == [4, 7, 8] and FORCING.of(FORK)[2] == {4: 4, 7: 7, 8: 8}


def test_the_safe_moves_are_every_safe_move_whether_or_not_forcing_ones_are_preferred(facts):
    assert facts.safe_moves(FORK) == facts.of(FORK)[1] == [4, 5, 6, 7, 8]
    assert FORCING.safe_moves(FORK) == [4, 5, 6, 7, 8] and FORCING.of(FORK)[1] == [4, 7, 8]
    assert DEEP.safe_moves(_ttt((0, 8), (4,), to_move=1)) == [1, 3, 5, 7]


def test_a_move_leaving_the_opponent_two_safe_replies_does_not_force():
    """X{0,1,5} O{2,3,7}, X to move: 4 and 8 each threaten one line (one block), 6 threatens nothing and leaves O two
    replies, both safe."""
    assert FORCING.of(_ttt((0, 1, 5), (2, 3, 7)))[1] == [4, 8]


def test_preferring_forcing_moves_keeps_every_safe_move_when_none_forces():
    empty = TicTacToe().initial_state()
    assert FORCING.of(empty)[1] == Facts(TicTacToe()).of(empty)[1] == list(range(9))


def test_a_move_that_ends_the_game_forces_nothing_and_a_reply_that_fills_the_board_counts_as_safe():
    """O to move, 3 and 6 empty: 6 wins at once (the game ends: not forcing); after 3 X's only reply fills the board
    and draws, one safe reply — so 3 forces."""
    assert FORCING.of(_ttt((0, 1, 4, 5), (2, 7, 8), to_move=1))[:2] == ([6], [3])


def test_preferring_forcing_moves_changes_neither_the_wins_nor_a_forced_safe_move():
    assert FORCING.of(_ttt((0, 1), (3, 4)))[0] == [2]
    assert FORCING.of(_ttt((0, 8), (4, 2)))[1] == [6]


def test_with_forcing_moves_preferred_the_levels_choose_among_them():
    assert choose(FORCING, FORK, {5: 0, 7: 1}, 2) == 7 and choose(Facts(TicTacToe()), FORK, {5: 0, 7: 1}, 2) == 5
    assert verify(FORCING, FORK, {4: 0, 5: 0}, 2)["won"]
    assert verify(Facts(TicTacToe()), FORK, {4: 0, 5: 0}, 2)["reason"] == "undefined"


TACTICS = Facts(TicTacToe(), ties="lowest", unlevelled="last", order="tactics")
WIN_IN_TWO = _ttt((0, 2), (1, 8))
LOSE_IN_TWO = _ttt((0, 2), (1,), to_move=1)


def test_ties_go_to_the_lowest_move_unless_told_otherwise(facts):
    assert facts.order == "lowest" and choose(LAST, WIN_IN_TWO, {}, 2) == 3
    with pytest.raises(ValueError, match="order"):
        Facts(TicTacToe(), ties="lowest", order="highest")


def test_an_order_needs_ties_to_break():
    with pytest.raises(ValueError, match="order"):
        Facts(TicTacToe(), order="tactics")


def test_a_move_wins_in_two_when_every_answer_leaves_an_immediate_win():
    """X{0,2} O{1,8}, X to move: only 6 makes two threats (0-3-6 and 2-4-6) that no answer blocks both of."""
    assert [a for a in TACTICS.of(WIN_IN_TWO)[1] if TACTICS._wins_in_two(WIN_IN_TWO, a)] == [6]


def test_a_move_ending_the_game_or_answered_by_one_doing_so_wins_nothing_in_two():
    """O to move, 3 and 6 empty: 6 wins at once (the game ends); after 3 X's only answer fills the board."""
    s = _ttt((0, 1, 4, 5), (2, 7, 8), to_move=1)
    assert not TACTICS._wins_in_two(s, 6) and not TACTICS._wins_in_two(s, 3)


def test_with_tactics_a_tie_goes_first_to_a_move_that_wins_in_two():
    assert choose(TACTICS, WIN_IN_TWO, {}, 2) == 6
    assert choose(TACTICS, _ttt((0, 2), (1, 5)), {}, 2) == 4


def test_with_tactics_a_tie_goes_next_to_a_move_that_does_not_lose_in_two():
    """X{0,2} O{1}, O to move: no move wins in two, and every safe move but the centre loses in two."""
    assert choose(LAST, LOSE_IN_TWO, {}, 2) == 3 and choose(TACTICS, LOSE_IN_TWO, {}, 2) == 4


def test_with_tactics_the_lowest_breaks_what_is_left():
    assert choose(TACTICS, WIN_IN_TWO, {3: 0, 7: 0}, 2) == 3
    assert choose(TACTICS, LOSE_IN_TWO, {5: 0, 7: 0, 8: 0}, 2) == 5


def test_with_tactics_only_the_deciding_level_is_ordered():
    assert choose(TACTICS, WIN_IN_TWO, {3: 0, 6: 0, 7: 0}, 2) == 6
    assert choose(TACTICS, WIN_IN_TWO, {3: 0, 7: 0, 6: 1}, 2) == 3
    assert choose(TACTICS, LOSE_IN_TWO, {3: 0, 4: 1}, 2) == 3


def test_tactics_change_neither_the_wins_nor_a_forced_safe_move_nor_a_level_of_one():
    assert choose(TACTICS, _ttt((0, 1), (3, 4)), {5: 0}, 2) == 2
    assert choose(TACTICS, _ttt((0, 8), (4, 2)), {3: 0}, 2) == 6
    assert choose(TACTICS, WIN_IN_TWO, {7: 0}, 2) == 7


def test_tactics_are_worked_out_once_per_position(monkeypatch):
    facts = Facts(TicTacToe(), ties="lowest", unlevelled="last", order="tactics")
    calls = []
    real = facts._loses_in_two
    monkeypatch.setattr(facts, "_loses_in_two", lambda s, a: calls.append(a) or real(s, a))
    choose(facts, LOSE_IN_TWO, {}, 2)
    first = len(calls)
    choose(facts, LOSE_IN_TWO, {5: 0, 7: 0}, 2)
    assert first == 6 and len(calls) == first


@pytest.mark.parametrize("order,state", [("lowest", LOSE_IN_TWO), ("tactics", _ttt((0, 8), (4, 2)))])
def test_tactics_are_worked_out_only_where_a_tie_can_need_them(monkeypatch, order, state):
    facts = Facts(TicTacToe(), ties="lowest", unlevelled="last", order=order)
    calls = []
    monkeypatch.setattr(facts, "_loses_in_two", lambda s, a: calls.append(a))
    monkeypatch.setattr(facts, "_wins_in_two", lambda s, a: calls.append(a))
    choose(facts, state, {}, 2)
    assert calls == []


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


SIX = _ttt((0, 1, 5), (2, 3, 7))
TWO = _ttt((0,), (1,))
PHASED = {5: 0, SWITCH: 6, phase_cell(4): 0}


def test_a_second_phase_s_cells_are_kept_apart_from_the_first_s():
    assert SWITCH == -1 and [phase_cell(c) for c in (0, 4, 8)] == [-2, -6, -10]


def test_a_switched_map_reads_its_first_levels_before_the_switch_and_its_second_from_it(facts):
    assert choose(facts, FORK, PHASED, 2) == 5 and facts.of(SIX)[1] == [4, 6, 8]
    assert choose(facts, SIX, PHASED, 2) == 4
    assert choose(facts, SIX, {**PHASED, SWITCH: 7}, 2) is None
    assert choose(facts, FORK, {**PHASED, SWITCH: 4}, 2) == 4


def test_a_map_without_a_switch_has_one_phase(facts):
    assert choose(facts, SIX, {5: 0, 4: 0}, 2) == 4 and choose(facts, FORK, {5: 0}, 2) == 5


def test_a_switched_map_wins_where_no_one_phase_map_of_one_level_does(facts):
    empty = facts.empty_cells(TWO)
    singles = [{c: 0 for c in cells} for r in range(len(empty) + 1) for cells in itertools.combinations(empty, r)]
    assert not any(verify(facts, TWO, m, 1)["won"] for m in singles)
    assert verify(facts, TWO, {3: 0, SWITCH: 4, phase_cell(4): 0}, 1)["won"]


def test_restricting_a_map_keeps_both_phases_cells_that_are_empty_and_the_switch(facts):
    assert restrict({5: 0, 0: 1, SWITCH: 6, phase_cell(4): 2, phase_cell(1): 0}, facts.empty_cells(FORK)) == {
        5: 0, SWITCH: 6, phase_cell(4): 2}
    assert restrict({5: 0, 0: 1}, facts.empty_cells(FORK)) == {5: 0}


def test_a_switched_map_pays_for_its_switch_and_its_second_phase_as_for_its_first():
    assert map_bits({4: 0, SWITCH: 6, phase_cell(4): 0, phase_cell(5): 1}, 5, 3) == 5 + 3 + 3 + 5 + 6
    assert map_bits({SWITCH: 6}, 30, 3) == 30 + 5 + 30
    with pytest.raises(ValueError, match="switch"):
        map_bits({4: 0, phase_cell(5): 1}, 5, 3)
    with pytest.raises(ValueError, match="levelled"):
        map_bits({SWITCH: 3, phase_cell(1): 0, phase_cell(2): 0}, 1, 3)


def test_simplify_drops_a_second_phase_the_win_does_not_need(facts):
    assert simplify(facts, FORK, {4: 0, SWITCH: 6, phase_cell(4): 0}, 2) == {4: 0}


def test_simplify_keeps_a_needed_second_phase_and_drops_its_unneeded_cells(facts):
    levels = {3: 0, SWITCH: 4, phase_cell(4): 0, phase_cell(8): 1}
    assert verify(facts, TWO, levels, 2)["won"] and not verify(facts, TWO, {3: 0}, 2)["won"]
    assert simplify(facts, TWO, levels, 2) == {3: 0, SWITCH: 4, phase_cell(4): 0}


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
