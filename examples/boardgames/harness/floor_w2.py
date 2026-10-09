"""§3.6 W2 — the whole-game projection redone with everything since W1: W1's 8 sampled frontier positions, in W1's
order, rebuilt with one long search for a pure steady state at each root (P2: 20 minutes at ply 8, 30 s below,
h199/h200), walks that honour the search's deadline, exceptions charged in walk order (h183), and maps shared: the 8
are built in two waves of 4, the second wave starting from a library of every map the first found. W1's 2-hour cap
and W1's projection (harness.floor_w1.w1_report: the opening plus the frontier times the sample's mean size, a lower
bound while any root is unfinished), judged against W1's 7.97M bits; and how much the second wave's leaves use the
library."""
from __future__ import annotations

from harness.floor_p2 import SPEC as P2
from harness.floor_w1 import SPEC as W1, w1_report

SPEC = {**W1, "measurement_fp": "4f5008ffc2ef",
        "indices": [362, 87, 316, 346, 230, 323, 393, 263],
        "wave_size": 4,
        "builder": {**P2["builder"], "seconds": 7200.0},
        "projection_support": 3_982_628,
        "projection_refute": 7_965_257,
        "library_support": 0.1}

w2_report = w1_report


def library_report(e: dict, spec: dict) -> dict:
    """The share of the leaves built with a library (the second wave's) that use one of its maps."""
    later = [r for r in e["roots"] if r["shared"] > 0]
    leaves = sum(r["bits"]["leaves"] for r in later)
    used = sum(r["library_leaves"] for r in later)
    if not later or not leaves:
        return {"verdict": "not_run"}
    share = used / leaves
    verdict = "supported" if share >= spec["library_support"] else "refuted" if used == 0 else "inconclusive"
    return {"verdict": verdict, "share": share, "used": used, "leaves": leaves}
