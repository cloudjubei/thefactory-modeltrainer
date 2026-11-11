"""Direct tests for harness/composed.py — a composed strategy, where a leaf's map plays only until the game reaches a
position with a node of its own, and `check`, which plays every line against every reply with no solver. Fixtures on
the 3x3 board: FORK, X{0,1} O{2,3}, X to move, which the empty map wins by the centre; LONE, X{5} O{6}, where only 8
wins and the empty map plays 0 and draws."""
from __future__ import annotations

from functools import lru_cache

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.composed import check
from harness.steady_state import Facts

GAME = TicTacToe()
LEVELS = 8
CAP = 100_000


def _facts():
    return Facts(GAME, ties="lowest", unlevelled="last")


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


FORK, LONE = _ttt((0, 1), (2, 3)), _ttt((5,), (6,))
KEY, LONE_KEY = GAME.state_key(FORK), GAME.state_key(LONE)


def _check(root, nodes, maps=None):
    return check(_facts(), root, {"nodes": nodes, "maps": maps or [{}]}, LEVELS, CAP)


def test_a_leaf_whose_map_wins_every_line_is_won():
    assert _check(FORK, {KEY: {"leaf": 0}}) == {"won": True, "reason": None}
    assert _check(LONE, {LONE_KEY: {"leaf": 1}}, [{}, {8: 0}])["won"]


def test_a_leaf_whose_map_loses_a_line_is_not_won():
    assert _check(LONE, {LONE_KEY: {"leaf": 0}}) == {"won": False, "reason": "lost"}
    assert _check(FORK, {KEY: {"leaf": 1}}, [{}, {5: 0}])["reason"] == "lost"


def _grandchildren(state, move):
    after = GAME.step(state, move)
    return [g for g in (GAME.step(after, b) for b in GAME.legal_actions(after)) if not GAME.is_terminal(g)]


def test_a_node_below_a_leaf_takes_over_from_its_map():
    from harness.self_proof import Prover
    open_ = _ttt((8,), (7,))
    leaf = {GAME.state_key(open_): {"leaf": 1}}
    assert _check(open_, leaf, [{}, {4: 0}])["reason"] == "lost"
    p = Prover(_facts(), LEVELS, CAP)
    below = {}
    for g in _grandchildren(open_, 4):
        assert p.prove(g) is True
        below.update(p.strategy(g)["nodes"])
    assert _check(open_, {**leaf, **below}, [{}, {4: 0}])["won"]
    assert _check(open_, {**leaf, **below}, [{}, {5: 0}])["won"] is False


def test_after_a_move_node_every_position_of_the_side_needs_a_node():
    assert _check(LONE, {LONE_KEY: {"move": 8}})["reason"] == "undefined"


def test_a_move_node_whose_move_is_not_legal_is_refused():
    with pytest.raises(ValueError, match="not a legal move"):
        _check(FORK, {KEY: {"move": 2}})


def test_a_map_that_gives_no_move_is_undefined():
    facts = Facts(GAME, ties="exact", unlevelled="none")
    assert check(facts, FORK, {"nodes": {KEY: {"leaf": 0}}, "maps": [{}]}, LEVELS, CAP)["reason"] == "undefined"


def test_a_walk_past_its_cap_stops():
    assert check(_facts(), FORK, {"nodes": {KEY: {"leaf": 0}}, "maps": [{}]}, LEVELS, 1)["reason"] == "cap"


def test_mirrored_readings_and_exceptions_are_not_part_of_a_composed_strategy():
    with pytest.raises(ValueError):
        _check(FORK, {KEY: {"leaf": 1, "mirrored": True}}, [{}, {4: 0}])
    with pytest.raises(ValueError):
        _check(FORK, {KEY: {"leaf": 0, "exceptions": {KEY: 4}}})


def test_a_position_reached_under_two_maps_is_played_by_each():
    root = _ttt((), ())
    first = GAME.step(root, 4)
    nodes = {GAME.state_key(root): {"move": 4}}
    for i, reply in enumerate(GAME.step(first, b) for b in GAME.legal_actions(first)):
        nodes[GAME.state_key(reply)] = {"leaf": 1 + i % 2}
    maps = [{}, {0: 0}, {8: 0}]
    alone = [_check(r, {GAME.state_key(r): n}, maps)["won"] for r, n in
             ((GAME.step(first, b), nodes[GAME.state_key(GAME.step(first, b))]) for b in GAME.legal_actions(first))]
    assert _check(root, nodes, maps)["won"] is all(alone)


@lru_cache(maxsize=None)
def _value(s) -> int:
    if GAME.is_terminal(s):
        w = GAME.winner(s)
        return 0 if w is None else 1 if w == GAME.current_player(s) else -1
    return max(-_value(GAME.step(s, a)) for a in GAME.legal_actions(s))


def _best(s):
    return min(a for a in GAME.legal_actions(s) if -_value(GAME.step(s, a)) == 1 or (
        GAME.is_terminal(GAME.step(s, a)) and GAME.winner(GAME.step(s, a)) == GAME.current_player(s)))


@pytest.fixture
def policies(monkeypatch):
    """Maps as explicit policies, {position key: move}, the minimax's best move elsewhere when "best" is set: the
    checker's own logic, which node decides and which map plays, apart from the rule."""
    import harness.composed as composed

    def choose(_facts, s, levels, _n_levels):
        move = levels.get(GAME.state_key(s))
        return _best(s) if move is None and levels.get("best") else move
    monkeypatch.setattr(composed, "choose", choose)


OPEN = _ttt((8,), (7,))
AFTER = GAME.step(OPEN, 2)
R1, R2 = GAME.step(AFTER, 0), GAME.step(AFTER, 3)
Q = GAME.step(GAME.step(R1, 4), 3)
K = GAME.state_key


def test_the_fixture_s_two_lines_meet_at_one_position_where_one_move_loses():
    assert Q == GAME.step(GAME.step(R2, 4), 0) and _value(Q) == 1 and -_value(GAME.step(Q, 1)) != 1
    assert all(-_value(GAME.step(s, a)) == 1 for s, a in ((OPEN, 2), (R1, 4), (R2, 4)))


def test_a_leaf_below_hands_play_to_its_own_map(policies):
    wrong = {"best": True, K(OPEN): 2, K(R1): 4, K(R2): 4, K(Q): 1}
    assert _check(OPEN, {K(OPEN): {"leaf": 1}}, [{}, wrong])["reason"] == "lost"
    assert _check(OPEN, {K(OPEN): {"leaf": 1}, K(Q): {"leaf": 2}}, [{}, wrong, {"best": True}])["won"]


def test_a_map_stops_at_a_move_node(policies):
    nodes = {K(OPEN): {"leaf": 1}, K(R1): {"move": 4}}
    assert _check(OPEN, nodes, [{}, {"best": True, K(OPEN): 2}])["reason"] == "undefined"


@pytest.mark.parametrize("first, second", [(1, 2), (2, 1)])
def test_a_position_two_maps_reach_is_played_by_each(policies, first, second):
    good, bad = {"best": True, K(R1): 4, K(R2): 4}, {"best": True, K(R1): 4, K(R2): 4, K(Q): 1}
    maps = [{}, good, bad]
    nodes = {K(OPEN): {"move": 2}}
    for b in GAME.legal_actions(AFTER):
        reply = GAME.step(AFTER, b)
        if not GAME.is_terminal(reply):
            nodes[K(reply)] = {"leaf": first if b == 0 else second if b == 3 else 1}
    assert _check(OPEN, nodes, maps)["reason"] == "lost"

