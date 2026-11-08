"""Direct tests for harness/level_context.py — harness.strategy_file with each level coded under what the decoder has
read of its map: the highest level so far, the cell's row and whether the cell below holds a level. Every other call is
strategy_file's own, and the strategy reads back from the bytes. Fixture X{0,1} O{2,3}, X to move, on the 3x3 board:
five safe moves, only the centre (4) wins."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.arithmetic_coding import Decoder, Encoder
from harness.level_context import LevelContext, decode_strategy, encode_strategy, level_context
from harness.steady_state import Facts
from harness.strategy_file import code_strategy, encode_strategy as plain_encode

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


TWO_MAPS = {"nodes": _tree(lambda i: {"leaf": 1 + i % 2}), "maps": [{}, {4: 0, 5: 1, 7: 2}, {6: 3, 8: 0}]}
EXCEPTED = {"nodes": {KEY: {"leaf": 1, "exceptions": {KEY: 4}}}, "maps": [{}, {4: 1, 5: 0}]}


class _Recording(Encoder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def symbol(self, family, context, n, value):
        self.calls.append((family, context, value))
        return super().symbol(family, context, n, value)


def _calls(source, wrap):
    io = _Recording()
    code_strategy(LevelContext(io, 3) if wrap else io, _facts(), FORK, LEVELS, CELLS, source)
    return io.calls


def _content(strategy):
    return {k: {**n, "leaf": strategy["maps"][n["leaf"]]} if "leaf" in n else n for k, n in strategy["nodes"].items()}


def test_a_level_reads_the_highest_level_so_far_its_row_and_the_cell_below():
    assert level_context(4, {}, 3) == (-1, 1, False)
    assert level_context(7, {1: 0, 4: 2}, 3) == (2, 2, True)
    assert level_context(5, {4: 0}, 3) == (0, 1, False)
    assert level_context(2, {0: 3}, 3) == (3, 0, False)


@pytest.mark.parametrize("source", [TWO_MAPS, EXCEPTED])
def test_the_strategy_reads_back_from_the_bytes(source):
    data, bits, _ = encode_strategy(_facts(), FORK, LEVELS, CELLS, source)
    decoded = decode_strategy(_facts(), FORK, LEVELS, CELLS, data)
    assert _content(decoded) == _content(source) and bits <= 8 * len(data)


def test_each_level_is_coded_under_its_context_and_each_map_starts_afresh():
    levels = [(context, value) for family, context, value in _calls(TWO_MAPS, True) if family == "levels"]
    assert levels == [((-1, 1, False), 0), ((0, 1, False), 1), ((1, 2, True), 2), ((-1, 2, False), 3),
                      ((3, 2, False), 0)]


def test_every_other_call_is_strategy_file_s_own():
    plain = [c for c in _calls(EXCEPTED, False) if c[0] != "levels"]
    wrapped = [c for c in _calls(EXCEPTED, True) if c[0] != "levels"]
    assert wrapped == plain and any(f == "exception_flags" for f, _, _ in plain)
    assert [v for f, _, v in _calls(TWO_MAPS, True) if f == "levels"] == [
        v for f, _, v in _calls(TWO_MAPS, False) if f == "levels"]


def test_levels_sharing_a_context_learn_from_each_other_and_nothing_else_changes():
    source = {"nodes": _tree(lambda i: {"leaf": 1 + i % 2}), "maps": [{}, {4: 0}, {5: 0}]}
    _, _, parts = encode_strategy(_facts(), FORK, LEVELS, CELLS, source)
    _, _, plain = plain_encode(_facts(), FORK, LEVELS, CELLS, source)
    assert {k: v for k, v in parts.items() if k != "levels"} == {k: v for k, v in plain.items() if k != "levels"}
    assert parts["levels"] < plain["levels"] == 6


def test_the_board_s_width_comes_from_the_game():
    data, _, _ = encode_strategy(_facts(), FORK, LEVELS, CELLS, TWO_MAPS)
    io = LevelContext(Decoder(data), GAME.board_shape[1])
    assert _content(code_strategy(io, _facts(), FORK, LEVELS, CELLS)) == _content(TWO_MAPS)


def test_a_decoder_reading_another_width_does_not_read_the_strategy_back():
    data, _, _ = encode_strategy(_facts(), FORK, LEVELS, CELLS, TWO_MAPS)
    try:
        wrong = _content(code_strategy(LevelContext(Decoder(data), 9), _facts(), FORK, LEVELS, CELLS))
    except (ValueError, KeyError, IndexError):
        return
    assert wrong != _content(TWO_MAPS)
