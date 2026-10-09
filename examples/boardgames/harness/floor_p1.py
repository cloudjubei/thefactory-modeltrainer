"""§3.6 P1 — bigger pure leaves, piloted: W1's three largest frontier strategies (#316 29,935 bits, #323 25,324, #230
unfinished at 26,062) rebuilt with leaves allowed from the frontier position itself and long searches for pure
steady states near it — 20 minutes at ply 8, 5 minutes at ply 10, W1's 30 seconds deeper — as WeakC4 covers whole
subtrees with one map (h192: a few big subtrees set the whole game's size). Everything else as W1, with a 3-hour cap.
Judged against W1 on the same positions (evidence/c49_w1.json.gz): the two strategies W1 finished at no more than half
their W1 size together, and all three complete. An unfinished strategy counts at its partial size, a lower bound:
enough to refute, never to support."""
from __future__ import annotations

SPEC = {
    "positions_fp": "bf548bd7612c",
    "measurement_fp": "223cfe3c6122",
    "ply": 8,
    "roots": 3,
    "seed": 4,
    "source": "opening",
    "frontier": 671,
    "indices": [316, 323, 230],
    "builder": {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 0, "reuse_window": 200,
                "seconds": 10800.0, "accept": 30, "budgets": [[0, 1200.0], [2, 300.0]]},
    "search": {"kind": "local", "seconds": 30.0, "cap": 1_000_000, "cache_limit": 1_500_000},
    "size_support": 0.5,
    "complete_support": 3,
    "complete_refute": 0,
}


def _integrity(e: dict, spec: dict, w1: dict) -> list[str]:
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
    before = {r["index"]: r for r in w1["roots"]}
    if any(r["index"] not in before or r["board"] != before[r["index"]]["board"] for r in e["roots"]):
        problems.append("a root is not the position W1 built under the same index")
    if not all(r["checked"] and r["moves_win"] and r["leaves_certified"] for r in e["roots"] if r["complete"]):
        problems.append("a completed strategy failed an independent check")
    return problems


def p1_report(e: dict, spec: dict, w1: dict) -> dict:
    integrity = _integrity(e, spec, w1)
    if integrity:
        return {"integrity": integrity, "size": {"verdict": "not_run"}, "complete": {"verdict": "not_run"}}
    before = {r["index"]: r for r in w1["roots"]}
    paired = [r for r in e["roots"] if before[r["index"]]["complete"]]
    p1 = sum(r["bits"]["nodes"] for r in paired)
    old = sum(before[r["index"]]["bits"]["nodes"] for r in paired)
    lower = not all(r["complete"] for r in paired)
    size = ("refuted" if p1 >= old else "supported" if not lower and p1 <= spec["size_support"] * old
            else "inconclusive")
    n = sum(1 for r in e["roots"] if r["complete"])
    complete = ("supported" if n >= spec["complete_support"] else "refuted" if n <= spec["complete_refute"]
                else "inconclusive")
    return {"integrity": [], "size": {"verdict": size, "p1": p1, "w1": old, "lower_bound": lower},
            "complete": {"verdict": complete, "count": n}}
