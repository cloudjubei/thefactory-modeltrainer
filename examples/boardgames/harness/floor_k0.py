"""§3.6 K0 — a pilot before K1: do two-phase maps (harness.steady_state SWITCH) make smaller leaves where one-phase ones
ran out? At up to 3 positions per root (seed 0) where L6's strategies for F1's 16 heaviest of L4's 32 split — no leaf
the builder would keep was found there; #297, one leaf in L6, has none — the 30-s local search runs without and with
`phases`, same seed and budget (scripts/c4_phase_pilot.py). Judged by sign over the positions where the two smallest
leaves differ (a search with no leaf counting as larger than any): supported when the two-phase search is smaller at
least twice as often as larger, refuted when it is smaller no more often than larger; undecidable with fewer than 10
differing."""
from __future__ import annotations

import math

from harness.floor_l4 import SPEC as L4

SPEC = {"measurement_fp": "54d2441217fd", "source": "c49_l6.json.gz", "roots": L4["indices"][:16], "per_root": 3,
        "seed": 0, "seconds": 30.0, "cap": 1_000_000, "cache_limit": 1_500_000, "support": 2, "min_differ": 10}


def k0_sign(e: dict, spec: dict) -> dict:
    if (e.get("measurement_fingerprint") != spec["measurement_fp"] or e.get("source") != spec["source"]
            or sorted({r["root"] for r in e["rows"]}) != sorted(set(spec["roots"]) - {297})):
        return {"verdict": "not_run"}

    def size(arm):
        return math.inf if arm["bits"] is None else arm["bits"]

    smaller = sum(1 for r in e["rows"] if size(r["two"]) < size(r["one"]))
    larger = sum(1 for r in e["rows"] if size(r["two"]) > size(r["one"]))
    if smaller + larger < spec["min_differ"]:
        verdict = "inconclusive"
    else:
        verdict = ("refuted" if smaller <= larger else "supported" if smaller >= spec["support"] * larger
                   else "inconclusive")
    return {"verdict": verdict, "smaller": smaller, "larger": larger,
            "switched": sum(1 for r in e["rows"] if r["two"]["switched"])}
