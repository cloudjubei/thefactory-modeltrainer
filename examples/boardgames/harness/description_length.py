"""The north star's "smallest total description" of a strategy, in bits (generic: any game, any table, any net).

A hybrid strategy is a net plus a table of exceptions. The net costs its parameters at a declared precision. Each
table entry names one position of the strategy's own tree and one move: ceil(log2(tree positions)) +
ceil(log2(actions)) bits. A table alone (every position tabled, no net) and a net alone (no entries) are the two
extremes of the same sum. Only certified strategies are ranked — a smaller description that is not perfect where it
claims to be does not count."""
from __future__ import annotations

import math


def entry_bits(positions: int, actions: int) -> int:
    """Bits to name one of `positions` tree positions and one of `actions` moves."""
    if positions < 1:
        raise ValueError(f"a table needs a tree of at least one position, got {positions} positions")
    if actions < 1:
        raise ValueError(f"a game needs at least one action, got {actions} actions")
    return math.ceil(math.log2(positions)) + math.ceil(math.log2(actions))


def hybrid_bits(params: int, bits_per_param: int, entries: int, positions: int, actions: int) -> dict:
    """{"net", "table", "total"} bits of a net of `params` parameters plus `entries` exceptions over a tree of
    `positions` positions."""
    if entries > positions:
        raise ValueError(f"a table cannot hold more entries ({entries}) than its tree has positions ({positions})")
    net = params * bits_per_param
    table = entries * entry_bits(positions, actions) if entries else 0
    return {"net": net, "table": table, "total": net + table}


def ranking(variants: dict) -> list:
    """(name, total bits) of the certified variants, smallest first."""
    return sorted(((name, v["total"]) for name, v in variants.items() if v["certified"]), key=lambda kv: kv[1])
