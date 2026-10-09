"""Direct tests for harness/opening.py — an opening for one side down to a fixed depth: at each of its positions a
winning move whose replies are most often already won by the empty map, every reply followed, and the side's
positions at the depth that still need a strategy collected. Tic-tac-toe fixtures."""
from __future__ import annotations

from games.tictactoe import TicTacToe, TTTState
from harness.opening import frontier
from harness.steady_state import Facts, verify

GAME = TicTacToe()


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


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


def _run(root, plies):
    return frontier(Facts(GAME), root, _winning, plies, n_levels=2, cap=100_000)


EARLY = _ttt((0,), (1,))


def test_every_reply_ends_in_a_win_a_trivial_position_or_the_frontier():
    r = _run(EARLY, 2)
    key = GAME.state_key(EARLY)
    assert set(r["moves"]) == {key} and r["moves"][key] in _winning([EARLY])[0]
    after = GAME.step(EARLY, r["moves"][key])
    trivial, front = set(r["trivial"]), {GAME.state_key(s) for s in r["frontier"]}
    for b in GAME.legal_actions(after):
        child = GAME.step(after, b)
        assert child.done or GAME.state_key(child) in trivial | front
    assert all(sum(1 for v in s.board if v) == 4 for s in r["frontier"])


def test_frontier_positions_still_need_a_strategy_and_trivial_ones_do_not():
    r = _run(EARLY, 2)
    facts = Facts(GAME)
    assert r["frontier"] and not any(verify(facts, s, {}, 2)["won"] for s in r["frontier"])
    states = {GAME.state_key(GAME.step(GAME.step(EARLY, r["moves"][GAME.state_key(EARLY)]), b)):
              GAME.step(GAME.step(EARLY, r["moves"][GAME.state_key(EARLY)]), b)
              for b in GAME.legal_actions(GAME.step(EARLY, r["moves"][GAME.state_key(EARLY)]))}
    assert all(verify(facts, states[k], {}, 2)["won"] for k in r["trivial"])


def test_the_move_with_the_most_trivially_won_replies_is_chosen():
    root = _ttt((0,), (5,))
    assert sorted(_winning([root])[0]) == [2, 4, 6]
    assert _run(root, 2)["moves"][GAME.state_key(root)] == 4


def test_a_depth_of_zero_returns_the_root_itself():
    r = _run(EARLY, 0)
    assert r["moves"] == {} and [GAME.state_key(s) for s in r["frontier"]] == [GAME.state_key(EARLY)]


def test_a_root_already_won_by_the_empty_map_needs_nothing():
    won = _ttt((0, 1), (3, 4))
    r = _run(won, 2)
    assert r == {"moves": {}, "trivial": [GAME.state_key(won)], "frontier": []}


def test_a_position_reached_twice_is_collected_once():
    r = _run(_ttt((4,), (1,)), 4)
    keys = [GAME.state_key(s) for s in r["frontier"]]
    assert len(keys) == len(set(keys)) and len(r["trivial"]) == len(set(r["trivial"]))


def test_a_root_its_side_cannot_win_is_refused():
    import pytest

    with pytest.raises(ValueError, match="cannot win"):
        _run(_ttt((), ()), 2)
