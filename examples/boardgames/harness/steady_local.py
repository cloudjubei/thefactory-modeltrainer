"""Local search over priority maps (§3.6 option 1), the proposal step WeakC4 used (a genetic search, then exact
checking): a map is scored by the size of the complete leaf it makes — its own bits plus the exceptions it needs over
every line from the root (harness.steady_exceptions.needed, with the search's oracle) — and hill-climbing changes one
cell's level at a time, keeping a change unless the leaf grows. Counting failing lines instead, and stopping at each,
rewards a map that gives no move at the root (h170). A leaf with no exceptions is a steady state; any leaf is checked
without the oracle by harness.steady_exceptions.verify_with."""
from __future__ import annotations

import math
import time

from harness.steady_exceptions import exception_bits, needed
from harness.steady_state import Facts, map_bits


def leaf_bits(facts: Facts, root, levels: dict, n_levels: int, level_bits: int, winning, cap: int) -> dict:
    """{"bits": map bits + exception bits (None when the walk outgrows `cap`), "exceptions", "own_positions"}."""
    r = needed(facts, root, levels, n_levels, winning, cap)
    if r["status"] == "cap":
        return {"bits": None, "exceptions": {}, "own_positions": r["own_positions"]}
    bits = (map_bits(levels, len(facts.empty_cells(root)), level_bits)
            + exception_bits(len(r["exceptions"]), r["own_positions"], facts.game.num_actions))
    return {"bits": bits, "exceptions": r["exceptions"], "own_positions": r["own_positions"]}


def _cost(leaf: dict) -> float:
    return math.inf if leaf["bits"] is None else leaf["bits"]


def local_search(facts: Facts, root, n_levels: int, level_bits: int, winning, rng, seconds: float, cap: int,
                 start: dict | None = None) -> dict:
    """Hill-climb from `start` (or the empty map) until the smallest leaf has no exceptions or `seconds` pass. Returns
    {"status": "found" (the smallest leaf is a steady state) | "budget", "levels", "exceptions", "bits" (None when
    every walk outgrew `cap`), "own_positions", "evaluations"}."""
    deadline = time.monotonic() + seconds
    cells = facts.empty_cells(root)
    current = dict(start or {})
    now = leaf_bits(facts, root, current, n_levels, level_bits, winning, cap)
    best, best_leaf, evaluations = dict(current), now, 1
    while (best_leaf["bits"] is None or best_leaf["exceptions"]) and time.monotonic() < deadline:
        cell = rng.choice(cells)
        trial = dict(current)
        level = rng.randrange(n_levels + 1)
        if level == n_levels:
            trial.pop(cell, None)
        else:
            trial[cell] = level
        leaf = leaf_bits(facts, root, trial, n_levels, level_bits, winning, cap)
        evaluations += 1
        if _cost(leaf) <= _cost(now):
            current, now = trial, leaf
            if _cost(now) < _cost(best_leaf):
                best, best_leaf = dict(current), now
    found = best_leaf["bits"] is not None and not best_leaf["exceptions"]
    return {"status": "found" if found else "budget", "levels": best, "exceptions": best_leaf["exceptions"],
            "bits": best_leaf["bits"], "own_positions": best_leaf["own_positions"], "evaluations": evaluations}
