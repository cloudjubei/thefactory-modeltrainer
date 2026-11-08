"""§3.6 T3 — harness.strategy_file with each map's levels coded in context. strategy_file codes a level by its cell
alone, and after T2 (harness.map_orders) a map's levels are the lowest that keep its moves: they run up from 0 without a
gap, so what a map has already said narrows what comes next. `LevelContext` wraps an Encoder or a Decoder and codes each
level under `level_context`: the highest level read so far in the map, the cell's row and whether the cell below it
holds a level, all of it read before the level on both sides. strategy_file reads a map's cells from cell 0 up, each
cell's level straight after its flag, so the wrapper starts a map afresh at cell 0's flag. Every other call is passed
on unchanged; the board's width is the game's `board_shape`."""
from __future__ import annotations

from harness.arithmetic_coding import Decoder, Encoder
from harness.steady_state import Facts
from harness.strategy_file import code_strategy


def level_context(cell: int, done: dict, cols: int) -> tuple:
    """`done` is {cell: level} of the map's cells read before `cell`."""
    return max(done.values(), default=-1), cell // cols, cell - cols in done


class LevelContext:
    def __init__(self, io, cols: int):
        self.io, self.cols, self.done = io, cols, {}

    def symbol(self, family: str, context, n: int, value=None) -> int:
        if family == "levels":
            value = self.io.symbol(family, level_context(context, self.done, self.cols), n, value)
            self.done[context] = value
            return value
        if family == "cells" and context[0] == 0:
            self.done = {}
        return self.io.symbol(family, context, n, value)

    def uniform(self, family: str, n: int, value=None) -> int:
        return self.io.uniform(family, n, value)


def encode_strategy(facts: Facts, root, n_levels: int, cells: list, source: dict) -> tuple:
    """(the file's bytes, its length in bits, each part's ideal bits)."""
    io = Encoder()
    code_strategy(LevelContext(io, facts.game.board_shape[1]), facts, root, n_levels, cells, source)
    data, bits = io.finish()
    return data, bits, io.parts


def decode_strategy(facts: Facts, root, n_levels: int, cells: list, data: bytes) -> dict:
    return code_strategy(LevelContext(Decoder(data), facts.game.board_shape[1]), facts, root, n_levels, cells)
