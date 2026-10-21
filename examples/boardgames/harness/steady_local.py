"""Local search over priority maps (§3.6 option 1), the proposal step WeakC4 used (a genetic search, then exact
checking): a map is scored by the size of the complete leaf it makes — its own bits plus the exceptions it needs over
every line from the root (harness.steady_exceptions.needed, with the search's oracle) — and hill-climbing changes one
cell's level at a time, keeping a change unless the leaf grows. Counting failing lines instead, and stopping at each,
rewards a map that gives no move at the root (h170). A leaf with no exceptions is a steady state; any leaf is checked
without the oracle by harness.steady_exceptions.verify_with. With `phases`, an eighth of the changes are to the map's
second phase (harness.steady_state SWITCH): switching one on (a copy of the first, from a random number of stones past
the root's), or moving or dropping a switch; and once a map has a switch, half of the level changes go to its second
phase. A change is judged without the fixed cost of having a second phase (its switch and its bitmap) — a copy plays as
the first phase does, so it would never be taken — while the best leaf is kept by its whole size."""
from __future__ import annotations

import math
import time

from harness.steady_exceptions import exception_bits, needed
from harness.steady_state import SWITCH, Facts, map_bits, phase_cell

SWITCH_SHARE = 0.125


def leaf_bits(facts: Facts, root, levels: dict, n_levels: int, level_bits: int, winning, cap: int,
              deadline: float | None = None, walker=needed) -> dict:
    """{"bits": map bits + exception bits (None when the walk outgrows `cap` or runs past `deadline`), "exceptions",
    "own_positions"}. `walker` is the leaf walk — harness.steady_exceptions.needed, or one with its contract (the
    C walk for Connect-4, harness.native_leaf)."""
    r = walker(facts, root, levels, n_levels, winning, cap, deadline)
    if r["status"] != "ok":
        return {"bits": None, "exceptions": {}, "own_positions": r["own_positions"]}
    bits = (map_bits(levels, len(facts.empty_cells(root)), level_bits)
            + exception_bits(len(r["exceptions"]), r["own_positions"], facts.game.num_actions))
    return {"bits": bits, "exceptions": r["exceptions"], "own_positions": r["own_positions"]}


def _cost(leaf: dict) -> float:
    return math.inf if leaf["bits"] is None else leaf["bits"]


def _switched(levels: dict, rng, stones: int, empty: int) -> dict:
    if SWITCH not in levels:
        first = {c: k for c, k in levels.items() if c >= 0}
        return {**first, SWITCH: stones + 1 + rng.randrange(empty - 1),
                **{phase_cell(c): k for c, k in first.items()}}
    if rng.random() < 0.5:
        return {**levels, SWITCH: stones + 1 + rng.randrange(empty - 1)}
    return {c: k for c, k in levels.items() if c >= 0}


def local_search(facts: Facts, root, n_levels: int, level_bits: int, winning, rng, seconds: float, cap: int,
                 start: dict | None = None, walker=needed, phases: bool = False) -> dict:
    """Hill-climb from `start` (or the empty map) until the smallest leaf has no exceptions or `seconds` pass — every
    walk, the first included, stops at the deadline, and a walk cut short has no size. Returns
    {"status": "found" (the smallest leaf is a steady state) | "budget", "levels", "exceptions", "bits" (None when
    every walk outgrew `cap`), "own_positions", "evaluations"}."""
    deadline = time.monotonic() + seconds
    cells = facts.empty_cells(root)
    current = dict(start or {})
    now = leaf_bits(facts, root, current, n_levels, level_bits, winning, cap, deadline, walker)
    best, best_leaf, evaluations = dict(current), now, 1
    stones = facts.cells - len(cells)
    phase = map_bits({SWITCH: stones + 1}, len(cells), level_bits) - len(cells)
    while (best_leaf["bits"] is None or best_leaf["exceptions"]) and time.monotonic() < deadline:
        if phases and len(cells) > 1 and rng.random() < SWITCH_SHARE:
            trial = _switched(current, rng, stones, len(cells))
        else:
            cell = rng.choice(cells)
            if phases and SWITCH in current and rng.random() < 0.5:
                cell = phase_cell(cell)
            trial = dict(current)
            level = rng.randrange(n_levels + 1)
            if level == n_levels:
                trial.pop(cell, None)
            else:
                trial[cell] = level
        leaf = leaf_bits(facts, root, trial, n_levels, level_bits, winning, cap, deadline, walker)
        evaluations += 1
        if _cost(leaf) - (phase if SWITCH in trial else 0) <= _cost(now) - (phase if SWITCH in current else 0):
            current, now = trial, leaf
            if _cost(now) < _cost(best_leaf):
                best, best_leaf = dict(current), now
    found = best_leaf["bits"] is not None and not best_leaf["exceptions"]
    return {"status": "found" if found else "budget", "levels": best, "exceptions": best_leaf["exceptions"],
            "bits": best_leaf["bits"], "own_positions": best_leaf["own_positions"], "evaluations": evaluations}
