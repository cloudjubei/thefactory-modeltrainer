"""Direct tests for harness/exception_context.py — harness.cell_context with each exception flag also coded in context:
how many exceptions the leaf's walk has read so far, up to two, and how many flags since the last one, on a log scale.
Every other call is cell_context's own, and the strategy reads back from the bytes. Fixtures on the 3x3 board: WALK, a
leaf at the empty board whose walk reads 14 flags and 4 exceptions; EXCEPTED, one exception at X{0,1} O{2,3}; TWO_MAPS,
a move there and two maps read by the leaves below it."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.arithmetic_coding import Decoder, Encoder
from harness.cell_context import CellContext, encode_strategy as cell_encode
from harness.exception_context import ExceptionContext, decode_strategy, encode_strategy, exception_context
from harness.level_context import LevelContext
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


EMPTY, FORK = GAME.initial_state(), _ttt((0, 1), (2, 3))
WALK = (EMPTY, {"nodes": {GAME.state_key(EMPTY): {"leaf": 1, "exceptions": {
    GAME.state_key(_ttt((4,), (2,))): 8, GAME.state_key(_ttt((2, 4), (1, 6))): 8,
    GAME.state_key(_ttt((2, 4), (5, 6))): 8, GAME.state_key(_ttt((2, 4, 7), (1, 6, 8))): 5}}},
    "maps": [{}, {4: 0, 2: 1, 6: 2}]})
WALK_FLAGS = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
EXCEPTED = (FORK, {"nodes": {GAME.state_key(FORK): {"leaf": 1, "exceptions": {GAME.state_key(FORK): 4}}},
                   "maps": [{}, {4: 1, 5: 0}]})


def _two_maps():
    after = GAME.step(FORK, 4)
    nodes = {GAME.state_key(FORK): {"move": 4}}
    for i, b in enumerate(GAME.legal_actions(after)):
        c = GAME.step(after, b)
        if not GAME.is_terminal(c):
            nodes[GAME.state_key(c)] = {"leaf": 1 + i % 2}
    return FORK, {"nodes": nodes, "maps": [{}, {1: 2, 4: 0, 7: 1}, {5: 3, 8: 0}]}


TWO_MAPS = _two_maps()


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


class _Echo:
    def __init__(self):
        self.calls = []

    def symbol(self, family, context, n, value=None):
        self.calls.append((family, context))
        return value


def _calls(fixture, flags_too):
    root, source = fixture
    io = _Recording()
    inner = ExceptionContext(io) if flags_too else io
    code_strategy(LevelContext(CellContext(inner, 3), 3), _facts(), root, LEVELS, CELLS, source)
    return io.calls


def _content(strategy):
    return {k: {**n, "leaf": strategy["maps"][n["leaf"]]} if "leaf" in n else n for k, n in strategy["nodes"].items()}


def test_a_flag_reads_the_exceptions_so_far_and_the_flags_since_the_last_one():
    assert [exception_context(0, gap) for gap in range(4)] == [(0, 0), (0, 1), (0, 1), (0, 2)]
    assert exception_context(1, 6) == (1, 2) and exception_context(1, 7) == (1, 3)
    assert exception_context(2, 0) == (2, 0) and exception_context(3, 0) == (2, 0) and exception_context(9, 2) == (2, 1)


@pytest.mark.parametrize("fixture", [WALK, EXCEPTED, TWO_MAPS])
def test_the_strategy_reads_back_from_the_bytes(fixture):
    root, source = fixture
    data, bits, _ = encode_strategy(_facts(), root, LEVELS, CELLS, source)
    decoded = decode_strategy(_facts(), root, LEVELS, CELLS, data)
    assert _content(decoded) == _content(source) and bits <= 8 * len(data)


def test_each_flag_is_coded_under_its_context():
    flags = [(context, value) for family, context, value in _calls(WALK, True) if family == "exception_flags"]
    plain = [(context, value) for family, context, value in _calls(WALK, False) if family == "exception_flags"]
    assert [value for _, value in flags] == [value for _, value in plain] == WALK_FLAGS
    assert [context for context, _ in flags] == [(0, 0), (0, 1), (0, 1), (1, 0), (1, 1), (1, 1), (2, 0), (2, 0),
                                                 (2, 1), (2, 1), (2, 2), (2, 0), (2, 1), (2, 1)]


def test_each_leaf_s_walk_starts_afresh():
    io = _Echo()
    wrapped = ExceptionContext(io)
    for family, value in [("excepted", 1), ("exception_flags", 0), ("exception_flags", 1), ("kind", 0),
                          ("exception_flags", 0), ("excepted", 0), ("excepted", 1), ("exception_flags", 0),
                          ("exception_flags", 0)]:
        assert wrapped.symbol(family, "given", 2, value) == value
    assert io.calls == [("excepted", "given"), ("exception_flags", (0, 0)), ("exception_flags", (0, 1)),
                        ("kind", "given"), ("exception_flags", (1, 0)), ("excepted", "given"), ("excepted", "given"),
                        ("exception_flags", (0, 0)), ("exception_flags", (0, 1))]


@pytest.mark.parametrize("fixture", [WALK, EXCEPTED])
def test_every_other_call_is_cell_context_s_own(fixture):
    plain = [c for c in _calls(fixture, False) if c[0] != "exception_flags"]
    wrapped = [c for c in _calls(fixture, True) if c[0] != "exception_flags"]
    assert wrapped == plain and any(f == "exception_moves" for f, _, _ in plain)


@pytest.mark.parametrize("fixture", [WALK, TWO_MAPS])
def test_only_the_exception_flags_cost_changes(fixture):
    root, source = fixture
    _, _, parts = encode_strategy(_facts(), root, LEVELS, CELLS, source)
    _, _, plain = cell_encode(_facts(), root, LEVELS, CELLS, source)
    assert {k: v for k, v in parts.items() if k != "exception_flags"} == {
        k: v for k, v in plain.items() if k != "exception_flags"}
    assert (parts.get("exception_flags") != plain.get("exception_flags")) == (fixture is WALK)


@pytest.mark.parametrize("fixture", [WALK, TWO_MAPS])
def test_the_board_s_width_comes_from_the_game(fixture):
    root, source = fixture
    data, _, _ = encode_strategy(_facts(), root, LEVELS, CELLS, source)
    width = GAME.board_shape[1]
    io = LevelContext(CellContext(ExceptionContext(Decoder(data)), width), width)
    assert _content(code_strategy(io, _facts(), root, LEVELS, CELLS)) == _content(source)
