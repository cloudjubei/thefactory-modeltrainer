"""§3.6 W3 — deciding W2's projection: the two of W1's sampled frontier positions W2 left unfinished in 2 hours (#316,
#230) rebuilt with W2's builder for up to 6 hours, each starting from a library of every map W2's builds found. With
W2's six finished strategies they give all 8 sample sizes, so the projection — the opening plus the 671 frontier
positions at the sample's mean (harness.floor_w1) — is exact when both finish and a lower bound otherwise. Judged
against the trained net (~660K bits): no larger is supported only when exact; a lower bound above it refutes."""
from __future__ import annotations

import statistics

from harness.floor_w2 import SPEC as W2

SPEC = {**{k: v for k, v in W2.items() if k not in ("wave_size", "library_support", "projection_support",
                                                     "projection_refute", "complete_support", "complete_refute")},
        "measurement_fp": "276ae6b8c7cf",
        "indices": [316, 230],
        "roots": 2,
        "library_from": "c49_w2.json.gz",
        "builder": {**W2["builder"], "seconds": 21600.0},
        "projection_line": 660_000}


def _integrity(e: dict, spec: dict, w2: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    if e.get("positions_fingerprint") != spec["positions_fp"]:
        problems.append(f"positions {e.get('positions_fingerprint')} are not the registered {spec['positions_fp']}")
    if e.get("builder") != spec["builder"]:
        problems.append("the strategies were built with other settings than registered")
    if (e.get("opening") or {}).get("frontier") != spec["frontier"]:
        problems.append(f"the opening's frontier is not the registered {spec['frontier']} positions")
    if sorted(r["index"] for r in e["roots"]) != sorted(spec["indices"]):
        problems.append(f"roots {[r['index'] for r in e['roots']]} are not the registered {spec['indices']}")
    before = {r["index"]: r for r in w2["roots"]}
    if any(r["index"] not in before or r["board"] != before[r["index"]]["board"] for r in e["roots"]):
        problems.append("a root is not the position W2 built under the same index")
    library = sum(len(r["found_maps"]) for r in w2["roots"])
    if any(r["shared"] != library for r in e["roots"]):
        problems.append(f"a build did not start from W2's library of {library} maps")
    if not all(r["checked"] and r["moves_win"] and r["leaves_certified"] for r in e["roots"] if r["complete"]):
        problems.append("a completed strategy failed an independent check")
    return problems


def w3_report(e: dict, spec: dict, w2: dict) -> dict:
    integrity = _integrity(e, spec, w2)
    if integrity:
        return {"integrity": integrity, "projection": {"verdict": "not_run"}, "complete": {"verdict": "not_run"}}
    redone = {r["index"]: r for r in e["roots"]}
    sample = [redone.get(r["index"], r) for r in w2["roots"]]
    opening = e["opening"]
    bits = (opening["moves"] * 4 + opening["trivial"]
            + opening["frontier"] * statistics.mean(r["bits"]["nodes"] for r in sample))
    lower = not all(r["complete"] for r in sample)
    verdict = ("refuted" if bits > spec["projection_line"] else "supported" if not lower else "inconclusive")
    n = sum(1 for r in e["roots"] if r["complete"])
    complete = "supported" if n == len(e["roots"]) else "refuted" if n == 0 else "inconclusive"
    return {"integrity": [], "projection": {"verdict": verdict, "bits": bits, "lower_bound": lower},
            "complete": {"verdict": complete, "count": n}}
