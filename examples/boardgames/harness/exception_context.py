"""§3.6 T5 — harness.cell_context with each exception flag also coded in context. harness.strategy_file codes whether
the rule is overruled at each of the leaf's own positions with a choice, all under one context, as the leaf's walk
reaches them. Exceptions come together: once a line leaves the rule it meets more of them, and a long run without one
makes the next rarer. `ExceptionContext` wraps an Encoder or a Decoder and codes each flag under `exception_context`:
how many exceptions the walk has read so far, up to `ONES`, and how many flags since the last one (since the walk began
when there is none), on a log scale, all of it read before the flag on both sides. A walk follows its leaf's
`excepted` flag, so the wrapper starts afresh there. It sits under harness.cell_context's wrappers, which code the
maps; every other call is passed on unchanged."""
from __future__ import annotations

from harness.arithmetic_coding import Decoder, Encoder
from harness.cell_context import CellContext
from harness.level_context import LevelContext
from harness.steady_state import Facts
from harness.strategy_file import code_strategy

ONES = 2


def exception_context(ones: int, gap: int) -> tuple:
    return min(ones, ONES), (gap + 1).bit_length() - 1


class ExceptionContext:
    def __init__(self, io):
        self.io, self.ones, self.gap = io, 0, 0

    def symbol(self, family: str, context, n: int, value=None) -> int:
        if family == "excepted":
            self.ones, self.gap = 0, 0
        if family != "exception_flags":
            return self.io.symbol(family, context, n, value)
        value = self.io.symbol(family, exception_context(self.ones, self.gap), n, value)
        self.ones, self.gap = (self.ones + 1, 0) if value else (self.ones, self.gap + 1)
        return value

    def uniform(self, family: str, n: int, value=None) -> int:
        return self.io.uniform(family, n, value)


def _coder(io, facts: Facts) -> LevelContext:
    cols = facts.game.board_shape[1]
    return LevelContext(CellContext(ExceptionContext(io), cols), cols)


def encode_strategy(facts: Facts, root, n_levels: int, cells: list, source: dict) -> tuple:
    """(the file's bytes, its length in bits, each part's ideal bits)."""
    io = Encoder()
    code_strategy(_coder(io, facts), facts, root, n_levels, cells, source)
    data, bits = io.finish()
    return data, bits, io.parts


def decode_strategy(facts: Facts, root, n_levels: int, cells: list, data: bytes) -> dict:
    return code_strategy(_coder(Decoder(data), facts), facts, root, n_levels, cells)
