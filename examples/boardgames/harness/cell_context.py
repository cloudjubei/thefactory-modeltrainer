"""§3.6 T4 — harness.level_context with each map's cell flags also coded in context. harness.strategy_file codes whether
a cell holds a level by the cell and whether it is empty at the leaf that first reads the map. A map levels the cells
its safe moves would fill, and the cells of a column fill from the bottom, so what the map has said of the cells below
narrows what it says next. `CellContext` wraps an Encoder or a Decoder and codes each flag under `cell_context`: the
cell, whether it is empty there, and for each of the two cells below it whether that cell is empty and whether it
holds a level, all of it read before the flag on both sides. strategy_file reads a map's flags from cell 0 up, so the
wrapper starts a map afresh at cell 0. It sits under harness.level_context's wrapper, which codes the levels; every
other call is passed on unchanged, and the board's width is the game's `board_shape`."""
from __future__ import annotations

from harness.arithmetic_coding import Decoder, Encoder
from harness.level_context import LevelContext
from harness.steady_state import Facts
from harness.strategy_file import code_strategy

BELOW = 2


def cell_context(cell: int, empty: bool, frame: dict, levelled: set, cols: int) -> tuple:
    """`frame` is {cell: empty} and `levelled` the levelled cells, of the map's cells read before `cell`."""
    under = tuple((frame[c], c in levelled) for c in (cell - k * cols for k in range(1, BELOW + 1)) if c >= 0)
    return cell, empty, under


class CellContext:
    def __init__(self, io, cols: int):
        self.io, self.cols, self.frame, self.levelled = io, cols, {}, set()

    def symbol(self, family: str, context, n: int, value=None) -> int:
        if family != "cells":
            return self.io.symbol(family, context, n, value)
        cell, empty = context
        if cell == 0:
            self.frame, self.levelled = {}, set()
        value = self.io.symbol(family, cell_context(cell, empty, self.frame, self.levelled, self.cols), n, value)
        self.frame[cell] = empty
        if value:
            self.levelled.add(cell)
        return value

    def uniform(self, family: str, n: int, value=None) -> int:
        return self.io.uniform(family, n, value)


def _coder(io, facts: Facts) -> LevelContext:
    cols = facts.game.board_shape[1]
    return LevelContext(CellContext(io, cols), cols)


def encode_strategy(facts: Facts, root, n_levels: int, cells: list, source: dict) -> tuple:
    """(the file's bytes, its length in bits, each part's ideal bits)."""
    io = Encoder()
    code_strategy(_coder(io, facts), facts, root, n_levels, cells, source)
    data, bits = io.finish()
    return data, bits, io.parts


def decode_strategy(facts: Facts, root, n_levels: int, cells: list, data: bytes) -> dict:
    return code_strategy(_coder(Decoder(data), facts), facts, root, n_levels, cells)
