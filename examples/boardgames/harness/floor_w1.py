"""§3.6 W1 — the whole-game first-player strategy, projected: an opening from the empty board to ply 8
(harness.opening, the label cache as its oracle) needs moves at 204 positions and leaves 671 first-player positions
at ply 8 to strategies; 8 of those, sampled with a seed, are built as S5 builds (leaves from ply 10, kept with
exceptions at >= 30x, walk-order accounting) within 2 hours each (scripts/c4_strategy_s2.py --spec harness.floor_w1).
Judged: completion of the sample; and the projected size of the whole strategy — the opening (a node bit and a 3-bit
move per move, a node bit per position it already wins) plus the frontier times the sample's mean size — against the
trained net (~660K bits; WeakC4's ~35-45K is the outside benchmark). An unfinished root counts at its partial size,
which makes the projection a lower bound: enough to refute, never to support."""
from __future__ import annotations

import statistics

SPEC = {
    "positions_fp": "bf548bd7612c",
    "measurement_fp": "bc95c6d34925",
    "ply": 8,
    "roots": 8,
    "seed": 4,
    "source": "opening",
    "frontier": 671,
    "builder": {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 2, "reuse_window": 200,
                "seconds": 7200.0, "accept": 30},
    "search": {"kind": "local", "seconds": 30.0, "cap": 1_000_000, "cache_limit": 1_500_000},
    "complete_support": 6,
    "complete_refute": 3,
    "projection_support": 660_000,
    "projection_refute": 6_600_000,
}


def _integrity(e: dict, spec: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    if e.get("positions_fingerprint") != spec["positions_fp"]:
        problems.append(f"positions {e.get('positions_fingerprint')} are not the registered {spec['positions_fp']}")
    if e.get("builder") != spec["builder"]:
        problems.append("the strategies were built with other settings than registered")
    if (e.get("opening") or {}).get("frontier") != spec["frontier"]:
        problems.append(f"the opening's frontier is not the registered {spec['frontier']} positions")
    if len(e["roots"]) != spec["roots"]:
        problems.append(f"{len(e['roots'])} roots, not {spec['roots']}")
    if not all(r["checked"] and r["moves_win"] and r["leaves_certified"] for r in e["roots"] if r["complete"]):
        problems.append("a completed strategy failed an independent check")
    return problems


def w1_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if integrity:
        return {"integrity": integrity, "complete": {"verdict": "not_run"}, "projection": {"verdict": "not_run"}}
    roots = e["roots"]
    n = sum(1 for r in roots if r["complete"])
    complete = ("supported" if n >= spec["complete_support"] else "refuted" if n <= spec["complete_refute"]
                else "inconclusive")
    opening = e["opening"]
    bits = (opening["moves"] * 4 + opening["trivial"]
            + opening["frontier"] * statistics.mean(r["bits"]["nodes"] for r in roots))
    lower = n < len(roots)
    verdict = ("refuted" if bits > spec["projection_refute"]
               else "supported" if not lower and bits <= spec["projection_support"] else "inconclusive")
    hours = opening["frontier"] * statistics.mean(r["build_seconds"] for r in roots) / 3600
    return {"integrity": [], "complete": {"verdict": complete, "count": n},
            "projection": {"verdict": verdict, "bits": bits, "lower_bound": lower, "hours": hours}}
