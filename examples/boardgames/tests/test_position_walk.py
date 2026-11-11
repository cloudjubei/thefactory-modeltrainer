"""Direct tests for harness/position_walk.py — harness.strategy_file's walk with each reference ranked by how alike the
leaf's board is to the boards each map was read at, and each exception flag also under the plies below its leaf. Every
other call is the T5 coder's own (harness.exception_context over strategy_file), the walk refuses what strategy_file
refuses, and the strategy reads back from the bytes. Fixtures on the 3x3 board: FORK, X{0,1} O{2,3} with five safe
moves; WALK, a leaf at the empty board whose walk reads 14 flags and 4 exceptions; LIKE, X at 4 and a leaf at each
reply, maps 1 and 2 in turn: O at 2 and O at 8 are the mirror images of O at 0 and O at 6, so their maps rank first
although another map was read last."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import harness.position_walk as position_walk
import harness.strategy_file as strategy_file
from games.tictactoe import TicTacToe, TTTState
from harness.arithmetic_coding import Decoder, Encoder
from harness.cell_context import CellContext
from harness.exception_context import _coder as t5_coder, encode_strategy as t5_encode
from harness.level_context import LevelContext
from harness.position_walk import code_strategy, decode_strategy, depth_context, encode_strategy, likeness, ranked
from harness.steady_state import Facts, choose

GAME = TicTacToe()
CELLS = [2, 1, 0, 5, 4, 3, 8, 7, 6]
LEVELS = 8


def _facts(**kwargs):
    return Facts(GAME, **{"ties": "lowest", "unlevelled": "last", **kwargs})


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


EMPTY, FORK = GAME.initial_state(), _ttt((0, 1), (2, 3))
KEY = GAME.state_key(FORK)


def _replies(s, move):
    after = GAME.step(s, move)
    return [c for c in (GAME.step(after, b) for b in GAME.legal_actions(after)) if not GAME.is_terminal(c)]


def _tree(root, move, leaf, maps):
    return root, {"nodes": {GAME.state_key(root): {"move": move},
                            **{GAME.state_key(c): leaf(i) for i, c in enumerate(_replies(root, move))}}, "maps": maps}


WALK = (EMPTY, {"nodes": {GAME.state_key(EMPTY): {"leaf": 1, "exceptions": {
    GAME.state_key(_ttt((4,), (2,))): 8, GAME.state_key(_ttt((2, 4), (1, 6))): 8,
    GAME.state_key(_ttt((2, 4), (5, 6))): 8, GAME.state_key(_ttt((2, 4, 7), (1, 6, 8))): 5}}},
    "maps": [{}, {4: 0, 2: 1, 6: 2}]})
EXCEPTED = (FORK, {"nodes": {KEY: {"leaf": 1, "exceptions": {KEY: 4}}}, "maps": [{}, {4: 1, 5: 0}]})
LIKE = _tree(EMPTY, 4, lambda i: {"leaf": 1 + i % 2}, [{}, {0: 1}, {8: 2}])
RENUMBERED = _tree(FORK, 4, lambda i: {"leaf": 2 - i % 2}, [{}, {5: 1}, {7: 3, 8: 0}])
BOTH_WAYS = _tree(FORK, 4, lambda i: {"leaf": 1, **({"mirrored": True} if i % 2 else {})}, [{}, {5: 1}])
MIRRORED = (FORK, {"nodes": {KEY: {"leaf": 1, "mirrored": True}}, "maps": [{}, {5: 1, 6: 0}]})
EMPTY_MAP = _tree(FORK, 4, lambda i: {"leaf": 0}, [{}])
FIXTURES = [WALK, EXCEPTED, LIKE, RENUMBERED, BOTH_WAYS, MIRRORED, EMPTY_MAP]


class _Recording(Encoder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def symbol(self, family, context, n, value):
        self.calls.append((family, context, value))
        return super().symbol(family, context, n, value)

    def uniform(self, family, n, value):
        self.calls.append((family, n, value))
        return super().uniform(family, n, value)


def _calls(fixture, t5=False):
    root, source = fixture
    io = _Recording()
    if t5:
        strategy_file.code_strategy(t5_coder(io, _facts()), _facts(), root, LEVELS, CELLS, source)
    else:
        code_strategy(position_walk._coder(io, _facts()), _facts(), root, LEVELS, CELLS, source)
    return io.calls


def _round_trip(fixture, facts=None):
    root, source = fixture
    facts = facts or _facts()
    data, bits, parts = encode_strategy(facts, root, LEVELS, CELLS, source)
    return decode_strategy(facts, root, LEVELS, CELLS, data), bits, parts, data


def _content(strategy):
    return {k: {**n, "leaf": strategy["maps"][n["leaf"]]} if "leaf" in n else n for k, n in strategy["nodes"].items()}


def _refused(root, source, error=ValueError, facts=None):
    with pytest.raises(error):
        encode_strategy(facts or _facts(), root, LEVELS, CELLS, source)


def test_the_plies_below_the_leaf_are_read_on_a_log_scale():
    assert [depth_context(p) for p in (0, 1, 2, 3, 6, 7)] == [0, 1, 1, 2, 2, 3]


def test_a_map_s_likeness_is_its_best_agreement_either_way_round():
    boards, mirrored = [np.array([1, 0, 2]), np.array([0, 0, 0])], [np.array([2, 0, 1]), np.array([0, 0, 0])]
    assert likeness(boards, mirrored, [1, 1], np.array([2, 0, 1]), 3).tolist() == [-1, 3, -1]
    assert likeness(boards, mirrored, [2, 1], np.array([1, 0, 0]), 3).tolist() == [-1, 2, 2]
    assert likeness([], [], [], np.array([1, 0, 2]), 2).tolist() == [-1, -1]


def test_the_most_alike_map_comes_first_and_ties_keep_move_to_front():
    assert ranked([3, 1, 2, 0], {3: 5, 1: 7, 2: 5, 0: -1}) == [1, 3, 2, 0]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_the_strategy_reads_back_from_the_bytes(fixture):
    decoded, bits, _, data = _round_trip(fixture)
    assert _content(decoded) == _content(fixture[1]) and bits <= 8 * len(data)


def test_maps_come_back_in_first_use_order():
    assert _round_trip(RENUMBERED)[0]["maps"] == [{}, {7: 3, 8: 0}, {5: 1}]


def test_a_map_read_both_ways_is_paid_once():
    assert len(_round_trip(BOTH_WAYS)[0]["maps"]) == 2


def test_a_position_the_map_leaves_undefined_reads_an_exception_with_no_flag():
    facts = _facts(ties="exact", unlevelled="none")
    assert choose(facts, FORK, {}, LEVELS) is None
    source = (FORK, {"nodes": {KEY: {"leaf": 0, "exceptions": {KEY: 4}}}, "maps": [{}]})
    decoded, _, parts, _ = _round_trip(source, facts)
    assert decoded["nodes"][KEY]["exceptions"] == {KEY: 4} and "exception_flags" not in parts


@pytest.mark.parametrize("fixture", FIXTURES)
def test_every_call_but_the_references_and_exception_flags_is_the_t5_coder_s(fixture):
    other = {"reference", "exception_flags"}
    assert [c for c in _calls(fixture) if c[0] not in other] == [c for c in _calls(fixture, True) if c[0] not in other]


def test_a_map_is_named_by_its_place_among_the_maps_ranked_by_likeness():
    names = [value for family, _, value in _calls(LIKE) if family == "reference"]
    assert names == [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert names != [value for family, _, value in _calls(LIKE, True) if family == "reference"]


def test_each_exception_flag_also_reads_the_plies_below_its_leaf():
    assert [c for f, c, _ in _calls(WALK) if f == "exception_flags"] == [
        (0, 0, 0), (0, 1, 1), (0, 1, 2), (1, 0, 1), (1, 1, 1), (1, 1, 2), (2, 0, 2), (2, 0, 2), (2, 1, 1), (2, 1, 1),
        (2, 2, 1), (2, 0, 2), (2, 1, 1), (2, 1, 1)]


@pytest.mark.parametrize("fixture", [WALK, LIKE])
def test_only_the_references_and_exception_flags_cost_changes(fixture):
    """LIKE's maps move between ranks 2 and 3, of one width: the same bits whichever is named."""
    root, source = fixture
    _, _, parts, _ = _round_trip(fixture)
    _, _, t5 = t5_encode(_facts(), root, LEVELS, CELLS, source)
    changed = {k for k in set(parts) | set(t5) if parts.get(k) != t5.get(k)}
    assert changed == ({"exception_flags"} if fixture is WALK else set())


def test_the_file_is_as_long_as_its_parts():
    _, bits, parts, _ = _round_trip(LIKE)
    assert abs(sum(parts.values()) - bits) < 4 and {"kind", "reference", "cells", "levels"} <= set(parts)


def test_the_board_s_width_comes_from_the_game():
    root, source = LIKE
    data = _round_trip(LIKE)[3]
    width = GAME.board_shape[1]
    io = LevelContext(CellContext(Decoder(data), width), width)
    assert _content(code_strategy(io, _facts(), root, LEVELS, CELLS)) == _content(source)


def test_a_position_reached_twice_is_read_once(monkeypatch):
    calls = []
    real = position_walk._leaf
    nodes, stack = {}, [EMPTY]
    while stack:
        s = stack.pop()
        if GAME.is_terminal(s) or GAME.state_key(s) in nodes:
            continue
        if s.to_move == 1:
            stack.extend(GAME.step(s, b) for b in GAME.legal_actions(s))
        elif sum(v != 0 for v in s.board) == 4:
            nodes[GAME.state_key(s)] = {"leaf": 0}
        else:
            nodes[GAME.state_key(s)] = {"move": _facts().safe_moves(s)[0]}
            stack.append(GAME.step(s, nodes[GAME.state_key(s)]["move"]))

    def counted(io, facts, s, *rest):
        calls.append(GAME.state_key(s))
        return real(io, facts, s, *rest)

    monkeypatch.setattr(position_walk, "_leaf", counted)
    _round_trip((EMPTY, {"nodes": nodes, "maps": [{}]}))
    assert set(Counter(calls).values()) == {2} and len(set(calls)) == sum("leaf" in n for n in nodes.values())


def test_a_leaf_without_exceptions_walks_nothing(monkeypatch):
    monkeypatch.setattr(position_walk, "_exceptions", lambda *a, **k: pytest.fail("walked a leaf with no exceptions"))
    _round_trip((FORK, {"nodes": {KEY: {"leaf": 0}}, "maps": [{}]}))


def test_a_move_that_is_not_safe_is_refused():
    _refused(FORK, {"nodes": {KEY: {"move": 2}}, "maps": [{}]})


def test_a_position_the_strategy_reaches_without_a_node_is_refused():
    _refused(FORK, {"nodes": {KEY: {"move": 4}}, "maps": [{}]}, KeyError)


def test_an_exception_on_a_forced_position_is_refused():
    after = GAME.step(GAME.step(FORK, 4), 5)
    key = GAME.state_key(after)
    _refused(after, {"nodes": {key: {"leaf": 0, "exceptions": {key: 8}}}, "maps": [{}]})


def test_an_exception_playing_the_map_s_own_move_is_refused():
    _refused(FORK, {"nodes": {KEY: {"leaf": 0, "exceptions": {KEY: choose(_facts(), FORK, {}, LEVELS)}}},
                    "maps": [{}]})


def test_an_undefined_position_with_no_exception_is_refused():
    _refused(FORK, {"nodes": {KEY: {"leaf": 0, "exceptions": {("x", 0): 1}}}, "maps": [{}]},
             facts=_facts(ties="exact", unlevelled="none"))


def test_the_empty_map_is_never_mirrored():
    _refused(FORK, {"nodes": {KEY: {"leaf": 0, "mirrored": True}}, "maps": [{}]})
    _refused(FORK, {"nodes": {KEY: {"move": 4}, **{GAME.state_key(c): {"leaf": 0, **({"mirrored": True} if i else {})}
                                                   for i, c in enumerate(_replies(FORK, 4))}}, "maps": [{}]})


def test_a_mirrored_leaf_with_exceptions_is_refused():
    _refused(FORK, {"nodes": {KEY: {"leaf": 1, "mirrored": True, "exceptions": {KEY: 4}}}, "maps": [{}, {5: 1}]})
