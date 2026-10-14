"""§3.6 A5 — spending the idle time. Builds, improved and improved with lookahead, finish well inside their 2-hour cap
(h246), and leaves below the root were searched for only 30 s. A5 builds A4's eight positions exactly as A4 and then,
with the builder's `research`, re-searches the costliest leaves with exceptions for 120 s each from their own maps in
the time left — keeping a new map, pure or with fewer exceptions, only where the whole strategy shrinks — and improves
with lookahead once more. Measured inside each build, as A4: how much further that cuts what the lookahead pass left
(supported at >= 10%, refuted under 2%; undecided while a position is unfinished)."""
from __future__ import annotations

from harness.floor_a4 import SPEC as A4, _checked

SPEC = {**{k: v for k, v in A4.items() if k not in ("further_support", "further_refute", "stuck_above")},
        "measurement_fp": "b2f5efcf2ac9",
        "builder": {**A4["builder"], "research": 120.0},
        "further_support": 0.10,
        "further_refute": 0.02}


def a5_further(e: dict, spec: dict) -> dict:
    if not _checked(e, spec):
        return {"verdict": "not_run"}
    if not all(r["complete"] for r in e["roots"]):
        return {"verdict": "inconclusive"}
    before = sum(r["improved"]["lookahead_bits"] for r in e["roots"])
    bits = sum(r["bits"]["nodes"] for r in e["roots"])
    cut = (before - bits) / before if before else 0
    verdict = ("supported" if cut >= spec["further_support"] else "refuted" if cut < spec["further_refute"]
               else "inconclusive")
    return {"verdict": verdict, "lookahead_bits": before, "bits": bits, "cut": cut}
