"""Direct tests for harness/cell_context.py — harness.level_context with each cell's flag also coded in context: the
cell, whether it is empty where the map is first read, and for each of the two cells below it whether that cell is empty
and whether it holds a level. Every other call is level_context's own, and the strategy reads back from the bytes.
Fixture X{0,1} O{2,3}, X to move, on the 3x3 board: five safe moves, only the centre (4) wins."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.arithmetic_coding import Decoder, Encoder
from harness.cell_context import CellContext, cell_context, decode_strategy, encode_strategy
from harness.level_context import LevelContext, encode_strategy as level_encode
from harness.steady_state import Facts
from harness.strategy_file import code_strategy

GAME = TicTacToe()
CELLS = [2, 1, 0, 5, 4, 3, 8, 7, 6]
LEVELS = 8


def _facts():
    return Facts(GAME, ties="lowest", unlevelled="last")


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


FORK = _ttt((0, 1), (2, 3))
KEY = GAME.state_key(FORK)


def _tree(leaf):
    """A move at FORK, and `leaf(i)` at each reply that does not end the game."""
    after = GAME.step(FORK, 4)
    nodes = {KEY: {"move": 4}}
    for i, b in enumerate(GAME.legal_actions(after)):
        c = GAME.step(after, b)
        if not GAME.is_terminal(c):
            nodes[GAME.state_key(c)] = leaf(i)
    return nodes


TWO_MAPS = {"nodes": _tree(lambda i: {"leaf": 1 + i % 2}), "maps": [{}, {1: 2, 4: 0, 7: 1}, {5: 3, 8: 0}]}
EXCEPTED = {"nodes": {KEY: {"leaf": 1, "exceptions": {KEY: 4}}}, "maps": [{}, {4: 1, 5: 0}]}


class _Recording(Encoder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def symbol(self, family, context, n, value):
        self.calls.append((family, context, value))
        return super().symbol(family, context, n, value)


def _calls(source, cells_too):
    io = _Recording()
    inner = CellContext(io, 3) if cells_too else io
    code_strategy(LevelContext(inner, 3), _facts(), FORK, LEVELS, CELLS, source)
    return io.calls


def _content(strategy):
    return {k: {**n, "leaf": strategy["maps"][n["leaf"]]} if "leaf" in n else n for k, n in strategy["nodes"].items()}


def test_a_flag_reads_its_cell_whether_it_is_empty_and_the_two_cells_below():
    assert cell_context(1, True, {0: False}, set(), 3) == (1, True, ())
    assert cell_context(4, False, {1: True}, {1}, 3) == (4, False, ((True, True),))
    assert cell_context(7, True, {1: False, 4: True}, {1}, 3) == (7, True, ((True, False), (False, True)))
    assert cell_context(8, True, {2: True, 5: False}, {5}, 3) == (8, True, ((False, True), (True, False)))
    assert cell_context(10, True, {1: True, 4: True, 7: False}, {1, 4}, 3) == (10, True, ((False, False), (True, True)))


@pytest.mark.parametrize("source", [TWO_MAPS, EXCEPTED])
def test_the_strategy_reads_back_from_the_bytes(source):
    data, bits, _ = encode_strategy(_facts(), FORK, LEVELS, CELLS, source)
    decoded = decode_strategy(_facts(), FORK, LEVELS, CELLS, data)
    assert _content(decoded) == _content(source) and bits <= 8 * len(data)


def test_each_flag_is_coded_under_its_context_and_each_map_starts_afresh():
    flags = [(context, value) for family, context, value in _calls(TWO_MAPS, True) if family == "cells"]
    plain = [(context, value) for family, context, value in _calls(TWO_MAPS, False) if family == "cells"]
    assert [value for _, value in flags] == [value for _, value in plain] and len(flags) == 18
    assert [context[:2] for context, _ in flags] == [context for context, _ in plain]
    first, second = flags[:9], flags[9:]
    empty = {c: e for (c, e), _ in plain[:9]}
    assert first[7][0] == (7, empty[7], ((empty[4], True), (empty[1], True)))
    assert first[4][0] == (4, empty[4], ((empty[1], True),))
    assert second[1][0][2] == () and second[8][0] == (8, plain[17][0][1], ((plain[14][0][1], True),
                                                                           (plain[11][0][1], False)))
    assert second[7][0] == (7, plain[16][0][1], ((plain[13][0][1], False), (plain[10][0][1], False)))


def test_every_other_call_is_level_context_s_own():
    plain = [c for c in _calls(EXCEPTED, False) if c[0] != "cells"]
    wrapped = [c for c in _calls(EXCEPTED, True) if c[0] != "cells"]
    assert wrapped == plain and any(f == "levels" for f, _, _ in plain)
    assert any(f == "exception_flags" for f, _, _ in plain)


def test_flags_sharing_a_context_learn_from_each_other_and_nothing_else_changes():
    _, _, parts = encode_strategy(_facts(), FORK, LEVELS, CELLS, TWO_MAPS)
    _, _, plain = level_encode(_facts(), FORK, LEVELS, CELLS, TWO_MAPS)
    assert {k: v for k, v in parts.items() if k != "cells"} == {k: v for k, v in plain.items() if k != "cells"}
    assert parts["cells"] != plain["cells"]


def test_the_board_s_width_comes_from_the_game():
    data, _, _ = encode_strategy(_facts(), FORK, LEVELS, CELLS, TWO_MAPS)
    width = GAME.board_shape[1]
    io = LevelContext(CellContext(Decoder(data), width), width)
    assert _content(code_strategy(io, _facts(), FORK, LEVELS, CELLS)) == _content(TWO_MAPS)


def test_the_levels_keep_level_context_s_context():
    source = {"nodes": _tree(lambda i: {"leaf": 1 + i % 2}), "maps": [{}, {4: 0}, {5: 0}]}
    _, _, parts = encode_strategy(_facts(), FORK, LEVELS, CELLS, source)
    _, _, plain = level_encode(_facts(), FORK, LEVELS, CELLS, source)
    assert parts["levels"] == plain["levels"] < 6
