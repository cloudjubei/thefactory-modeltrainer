"""Exact codes for a priority map (§3.6), given the cells empty at its position — the decoder replays the strategy and
knows them. The builder's code (harness.steady_state.map_bits) spends a bit on every empty cell and the full level
width on every levelled one. The rule reads only the levels' order (harness.steady_state.choose passes over empty
levels), so the levels in use can be renumbered 0..u-1 without changing a move. Codes:

  bitmap  — the builder's: one bit per empty cell, then each levelled cell's level in full;
  chosen  — how many cells are levelled and which (an enumerative code), then each level in full;
  ranks   — chosen cells, how many levels are used, then each cell's rank among them;
  counts  — chosen cells, how many levels are used, how many cells each holds (a composition), and which cells hold
            which (a multinomial index).

A map coded with the cheapest of these costs that code plus two bits naming it."""
from __future__ import annotations

import math
from collections import Counter

from harness.exception_coding import enumerative_bits

CODES = ("bitmap", "chosen", "ranks", "counts")
SELECTOR_BITS = 2


def _bits_for(n: int) -> int:
    """Bits to name one of n equally likely things."""
    return (n - 1).bit_length()


def ranked(levels: dict) -> dict:
    """The map with its used levels renumbered 0..u-1 in order — the same moves under the rule."""
    order = {k: i for i, k in enumerate(sorted(set(levels.values())))}
    return {c: order[k] for c, k in levels.items()}


def code_bits(code: str, levels: dict, empty: int, n_levels: int) -> int:
    """Bits for the map `levels` ({cell: level}) at a position with `empty` empty cells, in `code`."""
    if code not in CODES:
        raise ValueError(f"unknown code {code!r}; one of {CODES}")
    k = len(levels)
    if k > empty:
        raise ValueError(f"{k} levelled cells cannot exceed the {empty} empty cells")
    width = _bits_for(n_levels)
    if code == "bitmap":
        return empty + width * k
    cells = enumerative_bits(empty, k)
    if code == "chosen":
        return cells + width * k
    if not k:
        return cells
    counts = sorted(Counter(levels.values()).values())
    u = len(counts)
    if code == "ranks":
        return cells + width + k * _bits_for(u)
    which = math.factorial(k)
    for n in counts:
        which //= math.factorial(n)
    return cells + width + _bits_for(math.comb(k - 1, u - 1)) + _bits_for(which)


def best_bits(levels: dict, empty: int, n_levels: int) -> tuple[str, int]:
    """(the cheapest code, its bits plus the selector naming it)."""
    costs = {c: code_bits(c, levels, empty, n_levels) for c in CODES}
    code = min(CODES, key=lambda c: costs[c])
    return code, costs[code] + SELECTOR_BITS
