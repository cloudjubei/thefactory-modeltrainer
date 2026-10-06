"""§3.6 S2 — complete certified strategies for the first player from sampled ply-10 positions of the canonical exact
Connect-4 table, as table moves plus steady-state leaves (harness.strategy_builder), each within a time cap
(scripts/c4_strategy_s2.py). A completed strategy must pass the oracle-free walk (strategy_builder.check), every
table move must be a winning move by the exact solver, and every leaf with a non-empty map must certify.

Judged: completion — SUPPORTED when at least `complete_support` of the roots finish within the cap, REFUTED at
`complete_refute` or fewer; compression — the same strategy as a 3-bit-per-position table over the composite's
bits, median over completed roots, SUPPORTED at >= 10x, REFUTED below 1x."""
from __future__ import annotations

import statistics

SPEC = {
    "positions_fp": "a276d6c10259",
    "measurement_fp": "f429a42919a7",
    "ply": 10,
    "roots": 8,
    "seed": 2,
    "builder": {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 2, "reuse_window": 200,
                "seconds": 7200.0},
    "search": {"max_constraints": 20_000, "conflicts": 1_000_000, "seconds": 30.0, "cap": 1_000_000, "lines": 64},
    "complete_support": 6,
    "complete_refute": 3,
    "compression_support": 10,
    "compression_refute": 1,
}


def _integrity(e: dict, spec: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    if e.get("positions_fingerprint") != spec["positions_fp"]:
        problems.append(f"positions {e.get('positions_fingerprint')} are not the registered {spec['positions_fp']}")
    if e.get("builder") != spec["builder"]:
        problems.append("the strategies were built with other settings than registered")
    if len(e["roots"]) != spec["roots"]:
        problems.append(f"{len(e['roots'])} roots, not {spec['roots']}")
    done = [r for r in e["roots"] if r["complete"]]
    if not all(r["checked"] and r["moves_win"] and r["leaves_certified"] for r in done):
        problems.append("a completed strategy failed an independent check")
    return problems


def s2_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if integrity:
        return {"integrity": integrity, "complete": {"verdict": "not_run"}, "compression": {"verdict": "not_run"}}
    done = [r for r in e["roots"] if r["complete"]]
    n = len(done)
    complete = ("supported" if n >= spec["complete_support"] else "refuted" if n <= spec["complete_refute"]
                else "inconclusive")
    if not done:
        return {"integrity": [], "complete": {"verdict": complete, "count": n}, "compression": {"verdict": "not_run"}}
    values = [3 * r["own_positions"] / r["bits"]["nodes"] for r in done]
    median = statistics.median(values)
    compression = ("supported" if median >= spec["compression_support"]
                   else "refuted" if median < spec["compression_refute"] else "inconclusive")
    return {"integrity": [], "complete": {"verdict": complete, "count": n},
            "compression": {"verdict": compression, "median": median, "values": values}}
