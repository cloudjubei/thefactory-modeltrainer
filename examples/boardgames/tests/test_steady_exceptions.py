"""Direct tests for harness/steady_exceptions.py — a priority map plus exceptions: where the map's move on a line stops
winning (or it has none), a table move overrides it and the map resumes below. Fixture X{0,1} O{2,3}, X to move:
only the centre (4) wins, and after it every O reply leaves X an immediate win."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.steady_exceptions import choose_with, exception_bits, needed, patch, verify_with
from harness.steady_state import Facts

GAME = TicTacToe()


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


FORK = _ttt((0, 1), (2, 3))


def _winning(states):
    out = []
    for s in states:
        me = GAME.current_player(s)
        moves = set()
        for a in GAME.legal_actions(s):
            child = GAME.step(s, a)
            if (child.done and child.winner == me) or (not child.done and GAME.position_value(child) == -1):
                moves.add(a)
        out.append(moves)
    return out


def test_an_exception_overrides_the_map_and_the_map_plays_elsewhere():
    facts = Facts(GAME)
    assert choose_with(facts, FORK, {5: 0}, 2, {GAME.state_key(FORK): 4}) == 4
    assert choose_with(facts, FORK, {5: 0}, 2, {}) == 5


def test_the_empty_map_needs_one_exception_at_the_root():
    facts = Facts(GAME)
    r = patch(facts, FORK, {}, 2, _winning, cap=10_000, max_exceptions=5)
    assert r == {"status": "patched", "exceptions": {GAME.state_key(FORK): 4}}
    assert verify_with(facts, FORK, {}, 2, r["exceptions"], 10_000)["won"]


def test_a_map_failing_several_lines_is_patched_until_it_wins():
    facts = Facts(GAME)
    r = patch(facts, FORK, {5: 0}, 2, _winning, cap=10_000, max_exceptions=10)
    assert r["status"] == "patched" and r["exceptions"]
    assert verify_with(facts, FORK, {5: 0}, 2, r["exceptions"], 10_000)["won"]
    moves = [(GAME_STATE, m) for GAME_STATE, m in r["exceptions"].items()]
    assert all(m in _winning([TTTState(board=k[0], to_move=k[1], winner=None, done=False)])[0] for k, m in moves)


def test_a_map_needing_more_exceptions_than_allowed_is_refused():
    r = patch(Facts(GAME), FORK, {5: 0}, 2, _winning, cap=10_000, max_exceptions=0)
    assert r == {"status": "too_many", "exceptions": {}}


def test_a_root_its_side_cannot_win_is_refused():
    with pytest.raises(ValueError, match="cannot win"):
        patch(Facts(GAME), TicTacToe().initial_state(), {}, 2, _winning, cap=100_000, max_exceptions=50)


def test_a_walk_past_the_cap_is_reported():
    assert patch(Facts(GAME), FORK, {}, 2, _winning, cap=2, max_exceptions=5)["status"] == "cap"


def test_verify_with_rejects_a_losing_exception():
    r = verify_with(Facts(GAME), FORK, {4: 0}, 2, {GAME.state_key(FORK): 6}, 10_000)
    assert not r["won"] and r["reason"] in {"draw", "undefined"}


def test_exceptions_cost_an_index_into_the_leaf_s_positions_and_a_move():
    assert exception_bits(0, 100, 9) == 0
    assert exception_bits(3, 100, 9) == 3 * (7 + 4)
    assert exception_bits(1, 1, 7) == 3


EARLY = _ttt((0,), (1,))


def test_a_forced_draw_is_a_failure():
    drawn = _ttt((0, 2, 3, 7), (1, 4, 5, 6))
    assert verify_with(Facts(GAME), drawn, {}, 2, {}, 10_000) == {"won": False, "reason": "draw"}


def test_a_failure_behind_a_later_reply_is_found():
    assert not verify_with(Facts(GAME), EARLY, {2: 0, 4: 1}, 2, {}, 100_000)["won"]


def test_a_refused_patch_returns_no_exceptions_even_when_it_had_some():
    facts = Facts(GAME)
    assert len(patch(facts, EARLY, {2: 0, 3: 1}, 2, _winning, 100_000, 10)["exceptions"]) == 2
    assert patch(facts, EARLY, {2: 0, 3: 1}, 2, _winning, 100_000, 1) == {"status": "too_many", "exceptions": {}}


def test_a_winning_map_move_is_never_overridden():
    facts = Facts(GAME)
    r = patch(facts, EARLY, {3: 0, 2: 1}, 2, _winning, 100_000, 10)
    assert r["status"] == "patched" and len(r["exceptions"]) == 1 and GAME.state_key(EARLY) not in r["exceptions"]


def test_an_empty_leaf_costs_nothing():
    assert exception_bits(0, 0, 7) == 0


class _Counted:
    def __init__(self):
        self.asked = []

    def __call__(self, states):
        self.asked.extend(GAME.state_key(s) for s in states)
        return _winning(states)


def test_needed_walks_every_line_and_fixes_each_position_the_map_does_not_win():
    r = needed(Facts(GAME), FORK, {}, 2, _winning, 10_000)
    assert r == {"status": "ok", "exceptions": {GAME.state_key(FORK): 4}, "own_positions": 5}


def test_needed_asks_the_oracle_only_where_the_map_has_no_immediate_win():
    oracle = _Counted()
    needed(Facts(GAME), FORK, {}, 2, oracle, 10_000)
    assert set(oracle.asked) == {GAME.state_key(FORK)}


def test_needed_gives_a_winning_leaf_whose_every_exception_is_necessary():
    facts = Facts(GAME)
    for levels in ({2: 0, 3: 1}, {3: 0, 2: 1}, {}, {5: 1}):
        exc = needed(facts, EARLY, levels, 2, _winning, 100_000)["exceptions"]
        assert verify_with(facts, EARLY, levels, 2, exc, 100_000)["won"]
        for key in exc:
            assert not verify_with(facts, EARLY, levels, 2, {k: m for k, m in exc.items() if k != key}, 100_000)["won"]
    assert len(needed(facts, EARLY, {}, 2, _winning, 100_000)["exceptions"]) > 1


def test_needed_refuses_an_oracle_that_calls_a_losing_move_winning():
    late = _ttt((0, 1, 5), (2, 3, 6))

    def liar(states):
        return [{7} if GAME.state_key(s) == GAME.state_key(late) else w for s, w in zip(states, _winning(states))]

    with pytest.raises(RuntimeError, match="not won"):
        needed(Facts(GAME), late, {}, 2, liar, 10_000)


def test_needed_refuses_an_oracle_with_no_winning_move_below_the_root():
    def empty_below(states):
        return [w if GAME.state_key(s) == GAME.state_key(EARLY) else set() for s, w in zip(states, _winning(states))]

    with pytest.raises(RuntimeError, match="no winning move"):
        needed(Facts(GAME), EARLY, {}, 2, empty_below, 100_000)


def test_verify_with_reports_a_walk_past_the_cap():
    assert verify_with(Facts(GAME), FORK, {4: 0}, 2, {}, 2) == {"won": False, "reason": "cap"}


def test_needed_reports_a_walk_past_the_cap():
    assert needed(Facts(GAME), FORK, {}, 2, _winning, 2)["status"] == "cap"


def test_needed_refuses_a_root_its_side_cannot_win():
    with pytest.raises(ValueError, match="cannot win"):
        needed(Facts(GAME), TicTacToe().initial_state(), {}, 2, _winning, 100_000)


def test_dense_exceptions_cost_a_mask_over_the_leaf_s_positions():
    assert exception_bits(50, 100, 8) == 100 + 50 * 3
    assert exception_bits(5, 100, 8) == 5 * (7 + 3)


def test_a_patch_with_exactly_the_allowed_exceptions_is_kept():
    r = patch(Facts(GAME), EARLY, {2: 0, 3: 1}, 2, _winning, 100_000, 2)
    assert r["status"] == "patched" and len(r["exceptions"]) == 2
