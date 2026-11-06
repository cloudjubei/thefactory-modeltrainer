"""Direct tests for harness/strategy_file.py — a strategy as a file: one walk from the root reads every node, map and
exception, a decoder gets back the same strategy from the bytes and the game's rules alone, and `replay` then plays the
decoded strategy against every reply. Fixture X{0,1} O{2,3}, X to move: five safe moves, only the centre (4) wins, and
after it every O reply leaves X an immediate win."""
from __future__ import annotations

from collections import Counter

import pytest

import harness.strategy_file as strategy_file
from games.tictactoe import TicTacToe, TTTState
from harness.mirror_builder import board_image, mirrored_move
from harness.steady_state import Facts, choose
from harness.arithmetic_coding import Encoder
from harness.strategy_file import (code_strategy, decode_strategy, encode_strategy, leaf_states, replay, replay_leaf,
                                   trim_maps)

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


FORK = _ttt((0, 1), (2, 3))
KEY = GAME.state_key(FORK)
EMPTY = GAME.initial_state()
WRONG = {5: 0}


def _replies(s, move):
    after = GAME.step(s, move)
    return [GAME.step(after, b) for b in GAME.legal_actions(after)]


def _tree(leaf):
    """A move at FORK, and `leaf(i, reply)` at each reply."""
    nodes = {KEY: {"move": 4}}
    for i, c in enumerate(_replies(FORK, 4)):
        if not GAME.is_terminal(c):
            nodes[GAME.state_key(c)] = leaf(i, c)
    return nodes


def _losing():
    """5 at FORK, then the highest move: O takes the centre and X never blocks 2-4-6."""
    nodes, stack = {}, [FORK]
    while stack:
        s = stack.pop()
        if GAME.is_terminal(s) or GAME.state_key(s) in nodes:
            continue
        if s.to_move == 1:
            stack.extend(GAME.step(s, b) for b in GAME.legal_actions(s))
            continue
        move = 5 if s == FORK else max(GAME.legal_actions(s))
        nodes[GAME.state_key(s)] = {"move": move}
        stack.append(GAME.step(s, move))
    return nodes


def _round_trip(source, root=FORK, facts=None):
    facts = facts or _facts()
    data, bits, parts = encode_strategy(facts, root, LEVELS, CELLS, source)
    return decode_strategy(facts, root, LEVELS, CELLS, data), bits, parts


def _content(strategy):
    """Each node with its map's levels in place of the map's index, whichever order the maps are numbered in."""
    return {k: {**n, "leaf": strategy["maps"][n["leaf"]]} if "leaf" in n else n for k, n in strategy["nodes"].items()}


def test_moves_and_the_empty_map_come_back():
    source = {"nodes": _tree(lambda i, c: {"leaf": 0}), "maps": [{}]}
    assert _content(_round_trip(source)[0]) == _content(source)


def test_maps_come_back_in_first_use_order_however_the_source_numbers_them():
    source = {"nodes": _tree(lambda i, c: {"leaf": 2 - i % 2}), "maps": [{}, {5: 1}, {7: 3, 8: 0}]}
    decoded = _round_trip(source)[0]
    assert _content(decoded) == _content(source)
    assert decoded["maps"] == [{}, {7: 3, 8: 0}, {5: 1}]


def test_a_map_is_paid_once_however_many_leaves_play_it():
    once = _round_trip({"nodes": _tree(lambda i, c: {"leaf": 1 if i == 0 else 0}), "maps": [{}, {5: 1, 7: 2}]})
    every = _round_trip({"nodes": _tree(lambda i, c: {"leaf": 1}), "maps": [{}, {5: 1, 7: 2}]})
    assert every[2]["cells"] == once[2]["cells"] and every[2]["levels"] == once[2]["levels"]


def test_a_level_of_zero_is_a_level():
    source = {"nodes": {KEY: {"leaf": 1}}, "maps": [{}, {4: 0}]}
    assert _round_trip(source)[0]["maps"][1] == {4: 0}


def test_a_map_may_level_a_cell_full_at_the_leaf_that_first_plays_it():
    source = {"nodes": {KEY: {"leaf": 1}}, "maps": [{}, {0: 2, 4: 1}]}
    assert _round_trip(source)[0]["maps"][1] == {0: 2, 4: 1}


def test_the_move_is_read_among_the_safe_moves():
    source = {"nodes": {KEY: {"move": 2}}, "maps": [{}]}
    with pytest.raises(ValueError):
        encode_strategy(_facts(), FORK, LEVELS, CELLS, source)


def test_a_legal_move_that_is_not_safe_is_refused():
    s = _ttt((0, 8), (2, 4))
    key = GAME.state_key(s)
    assert _facts().safe_moves(s) == [6] and 1 in GAME.legal_actions(s)
    with pytest.raises(ValueError):
        encode_strategy(_facts(), s, LEVELS, CELLS, {"nodes": {key: {"move": 1}}, "maps": [{}]})


def test_the_only_safe_move_costs_no_move_bits():
    s = _ttt((0, 8), (2, 4))
    after = GAME.step(s, 6)
    nodes = {GAME.state_key(s): {"move": 6}, **{GAME.state_key(GAME.step(after, b)): {"leaf": 0}
                                                for b in GAME.legal_actions(after)}}
    assert "move" not in _round_trip({"nodes": nodes, "maps": [{}]}, root=s)[2]


class _Recording(Encoder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def symbol(self, family, context, n, value):
        self.calls.append((family, context, value))
        return super().symbol(family, context, n, value)


def _calls(source, root=FORK):
    io = _Recording()
    code_strategy(io, _facts(), root, LEVELS, CELLS, source)
    return io.calls


def test_a_cell_s_context_is_whether_the_first_leaf_has_it_empty():
    cells = {c[0]: c[1] for f, c, _ in _calls({"nodes": {KEY: {"leaf": 1}}, "maps": [{}, {5: 1}]}) if f == "cells"}
    assert len(cells) == 9 and {i for i, empty in cells.items() if empty} == {4, 5, 6, 7, 8}


def test_a_mirrored_map_s_first_leaf_is_read_on_the_mirrored_board():
    source = {"nodes": {KEY: {"leaf": 1, "mirrored": True}}, "maps": [{}, {5: 1}]}
    cells = {c[0]: c[1] for f, c, _ in _calls(source) if f == "cells"}
    assert {i for i, empty in cells.items() if empty} == {3, 4, 6, 7, 8}


def test_the_map_played_last_is_the_cheapest_to_name():
    leaves = [c for c in _replies(FORK, 4) if not GAME.is_terminal(c)]
    plays = iter([1, 2, 2, 1])
    source = {"nodes": {KEY: {"move": 4}, **{GAME.state_key(c): {"leaf": next(plays)} for c in leaves}},
              "maps": [{}, {5: 1}, {7: 1}]}
    widths = [v for f, _, v in _calls(source) if f == "reference"]
    assert widths == [0, 0, 1, 1]


def test_a_position_the_strategy_reaches_without_a_node_is_refused():
    with pytest.raises(KeyError):
        encode_strategy(_facts(), FORK, LEVELS, CELLS, {"nodes": {KEY: {"move": 4}}, "maps": [{}]})


def test_a_position_reached_twice_is_read_once():
    calls = []
    real = strategy_file._leaf
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

    strategy_file._leaf = counted
    try:
        _round_trip({"nodes": nodes, "maps": [{}]}, root=EMPTY)
    finally:
        strategy_file._leaf = real
    assert set(Counter(calls).values()) == {2} and len(set(calls)) == sum("leaf" in n for n in nodes.values())


def test_exceptions_come_back_and_are_paid_only_where_a_decoder_reads_one():
    source = {"nodes": {KEY: {"leaf": 1, "exceptions": {KEY: 4}}}, "maps": [{}, WRONG]}
    decoded, _, parts = _round_trip(source)
    assert decoded["nodes"][KEY]["exceptions"] == {KEY: 4} and "exception_moves" in parts and "exception_flags" in parts


def test_a_leaf_without_exceptions_walks_nothing(monkeypatch):
    monkeypatch.setattr(strategy_file, "_exceptions", lambda *a, **k: pytest.fail("walked a leaf with no exceptions"))
    _round_trip({"nodes": {KEY: {"leaf": 0}}, "maps": [{}]})


def test_a_position_the_map_leaves_undefined_reads_an_exception_with_no_flag():
    facts = _facts(ties="exact", unlevelled="none")
    assert choose(facts, FORK, {}, LEVELS) is None
    decoded, _, parts = _round_trip({"nodes": {KEY: {"leaf": 0, "exceptions": {KEY: 4}}}, "maps": [{}]}, facts=facts)
    assert decoded["nodes"][KEY]["exceptions"] == {KEY: 4} and "exception_flags" not in parts


def test_an_undefined_position_with_no_exception_is_refused():
    facts = _facts(ties="exact", unlevelled="none")
    with pytest.raises(ValueError):
        encode_strategy(facts, FORK, LEVELS, CELLS, {"nodes": {KEY: {"leaf": 0, "exceptions": {("x", 0): 1}}},
                                                     "maps": [{}]})


def test_an_exception_on_a_forced_position_is_refused():
    after = _replies(FORK, 4)[0]
    key = GAME.state_key(after)
    with pytest.raises(ValueError):
        encode_strategy(_facts(), after, LEVELS, CELLS, {"nodes": {key: {"leaf": 0, "exceptions": {key: 8}}},
                                                         "maps": [{}]})


def test_an_exception_playing_the_map_s_own_move_is_refused():
    facts = _facts()
    with pytest.raises(ValueError):
        encode_strategy(facts, FORK, LEVELS, CELLS, {"nodes": {KEY: {"leaf": 0, "exceptions": {
            KEY: choose(facts, FORK, {}, LEVELS)}}}, "maps": [{}]})


def test_a_mirrored_leaf_comes_back_mirrored():
    source = {"nodes": {KEY: {"leaf": 1, "mirrored": True}}, "maps": [{}, {5: 1, 6: 0}]}
    decoded = _round_trip(source)[0]
    assert decoded["nodes"][KEY]["mirrored"] is True and decoded["maps"][1] == {5: 1, 6: 0}


def test_a_map_can_be_read_both_ways_and_is_paid_once():
    source = {"nodes": _tree(lambda i, c: {"leaf": 1, **({"mirrored": True} if i % 2 else {})}), "maps": [{}, {5: 1}]}
    decoded = _round_trip(source)[0]
    assert _content(decoded) == _content(source) and len(decoded["maps"]) == 2


def test_the_empty_map_is_never_mirrored():
    with pytest.raises(ValueError):
        encode_strategy(_facts(), FORK, LEVELS, CELLS, {"nodes": {KEY: {"leaf": 0, "mirrored": True}}, "maps": [{}]})


def test_a_mirrored_leaf_with_exceptions_is_refused():
    with pytest.raises(ValueError):
        encode_strategy(_facts(), FORK, LEVELS, CELLS, {
            "nodes": {KEY: {"leaf": 1, "mirrored": True, "exceptions": {KEY: 4}}}, "maps": [{}, {5: 1}]})


def test_the_file_is_as_long_as_its_parts():
    source = {"nodes": _tree(lambda i, c: {"leaf": 1 + i % 2}), "maps": [{}, {5: 1, 7: 2}, {6: 0, 8: 7}]}
    _, bits, parts = _round_trip(source)
    assert abs(sum(parts.values()) - bits) < 4 and {"kind", "reference", "cells", "levels"} <= set(parts)


def test_leaf_states_walks_the_nodes_and_stops_at_leaves():
    nodes = _tree(lambda i, c: {"leaf": 0})
    found = leaf_states(GAME, FORK, nodes)
    assert [GAME.state_key(s) for s, _ in found["leaves"]] == [k for k, n in nodes.items() if "leaf" in n]
    assert found["won"] is True


def test_leaf_states_finds_a_lost_line():
    assert leaf_states(GAME, FORK, _losing())["won"] is False


def test_trimming_drops_cells_full_at_every_leaf_that_plays_the_map_and_no_other():
    source = {"nodes": _tree(lambda i, c: {"leaf": 1}), "maps": [{}, {0: 3, 5: 1, 7: 2}]}
    assert trim_maps(_facts(), FORK, CELLS, source)["maps"] == [{}, {5: 1, 7: 2}]
    first, second = [c for c in _replies(FORK, 4) if not GAME.is_terminal(c)][:2]
    gone = next(i for i in range(9) if first.board[i] and not second.board[i])
    source = {"nodes": _tree(lambda i, c: {"leaf": 1}), "maps": [{}, {gone: 3}]}
    assert trim_maps(_facts(), FORK, CELLS, source)["maps"][1] == {gone: 3}


def test_trimming_reads_a_mirrored_leaf_on_the_mirrored_board():
    full = next(i for i in range(9) if FORK.board[i] and not FORK.board[CELLS[i]])
    source = {"nodes": {KEY: {"leaf": 1, "mirrored": True}}, "maps": [{}, {full: 3}]}
    assert trim_maps(_facts(), FORK, CELLS, source)["maps"][1] == {full: 3}
    source["nodes"][KEY] = {"leaf": 1}
    assert trim_maps(_facts(), FORK, CELLS, source)["maps"][1] == {}


def test_trimming_keeps_the_strategy_s_nodes_and_map_numbers():
    source = {"nodes": _tree(lambda i, c: {"leaf": 2}), "maps": [{}, {4: 1}, {5: 1}]}
    out = trim_maps(_facts(), FORK, CELLS, source)
    assert out["nodes"] == source["nodes"] and out["maps"][0] == {} and len(out["maps"]) == 3
    assert out["maps"][1] == {} and out["maps"][2] == {5: 1}


def test_a_decoded_winning_strategy_replays_as_won():
    source = {"nodes": _tree(lambda i, c: {"leaf": 0}), "maps": [{}]}
    decoded = _round_trip(source)[0]
    assert replay(_facts(), FORK, decoded, LEVELS, 1_000, CELLS) == {"won": True, "leaves": 4, "reason": None}


def test_a_lost_line_above_the_leaves_fails_the_replay():
    r = replay(_facts(), FORK, {"nodes": _losing(), "maps": [{}]}, LEVELS, 1_000, CELLS)
    assert r["won"] is False and r["reason"] == "lost line above the leaves"


def test_a_leaf_that_loses_fails_the_replay():
    strategy = {"nodes": {KEY: {"leaf": 1}}, "maps": [{}, WRONG]}
    r = replay(_facts(), FORK, strategy, LEVELS, 1_000, CELLS)
    assert r["won"] is False and r["reason"] == "a leaf failed"


def test_an_exception_can_save_a_leaf():
    strategy = {"nodes": {KEY: {"leaf": 1, "exceptions": {KEY: 4}}}, "maps": [{}, WRONG]}
    assert replay(_facts(), FORK, strategy, LEVELS, 1_000, CELLS)["won"] is True


def test_a_mirrored_leaf_is_played_on_the_mirrored_board():
    facts = _facts()
    levels = {4: 0}
    image = board_image(CELLS)(FORK)
    assert mirrored_move(facts, CELLS, FORK, levels, LEVELS) == 4 and choose(facts, image, levels, LEVELS) == 4
    assert replay_leaf(facts, FORK, {"leaf": 1, "mirrored": True}, [{}, levels], LEVELS, 1_000, CELLS) is True
    only_plain = {3: 0}
    assert replay_leaf(facts, FORK, {"leaf": 1}, [{}, only_plain], LEVELS, 1_000, CELLS) is True
    assert replay_leaf(facts, FORK, {"leaf": 1, "mirrored": True}, [{}, only_plain], LEVELS, 1_000, CELLS) is False


def test_replay_can_take_its_leaves_from_a_worker():
    strategy = {"nodes": _tree(lambda i, c: {"leaf": 0}), "maps": [{}]}
    assert replay(_facts(), FORK, strategy, LEVELS, 1_000, CELLS, check=lambda leaves: len(leaves) == 4)["won"] is True
    assert replay(_facts(), FORK, strategy, LEVELS, 1_000, CELLS, check=lambda leaves: False)["won"] is False
