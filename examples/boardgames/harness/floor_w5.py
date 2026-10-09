"""§3.6 W5 — the design P2 meant and never ran (h216): one 20-minute search for a pure steady state at each frontier
root and 30 s at every depth below ([[0, 1200], [2, 30]] — the second entry closes the first), with the leaves walked
in C (harness.native_leaf: lazy solves, keys built when read; h210, h214). Otherwise W2: W1's 8 sampled positions in
W1's order, two waves of 4 sharing a map library, a 2-hour cap, W1's projection (harness.floor_w1.w1_report). Judged on
completion, on the projection against W3's 1.62M-bit lower bound, and on total build time against W2's."""
from __future__ import annotations

from harness.floor_w2 import SPEC as W2, library_report
from harness.floor_w1 import w1_report

SPEC = {**W2, "measurement_fp": "bea9663e4304",
        "builder": {**W2["builder"], "budgets": [[0, 1200.0], [2, 30.0]]},
        "search": {**W2["search"], "walker": "native"},
        "projection_support": 1_618_856,
        "projection_refute": 1_618_856,
        "w2_build_seconds": 32_232.0,
        "time_support": 0.5}

w5_report = w1_report
w5_library = library_report


def time_report(e: dict, spec: dict) -> dict:
    total = sum(r["build_seconds"] for r in e["roots"])
    ratio = total / spec["w2_build_seconds"]
    return {"verdict": "supported" if ratio <= spec["time_support"] else "refuted" if ratio >= 1 else "inconclusive",
            "seconds": total, "ratio": ratio}
