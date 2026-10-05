"""Net cost against table cost as the certified depth grows (§C.49, after H3). Through a depth D the first player's
strategy tree holds every first-player position before ply D. A complete table is one move per position, read in the
tree's canonical walk order, so it needs no index: entries x ceil(log2 actions) bits. A hybrid is a net plus an
exception wherever the net's move is not optimal, stored as the cheaper of a sparse list (index + move) or a mask (one
bit per position + a move per exception). The break-even precision at a depth is the bits per weight at which net and
exceptions cost exactly the table: above it the table is smaller. It assumes the net keeps its moves at that
precision, so it bounds what quantization would have to achieve.

The judge reads a run's nets (scripts/c4_depth_cost.py): whether the net's exception share falls from an early to a
late ply of its own tree (pooled over the deep nets), and whether every deep net breaks even above a registered
precision at the judged depth."""
from __future__ import annotations

import math

SPEC = {
    "run": "c49_H3_deep",
    "era": "33939d5e2d76",
    "measurement_fp": None,
    "seeds": [481, 482, 483, 484, 485, 486, 487],
    "deep": [481, 482, 483],
    "depths": [9, 11, 13],
    "early_ply": 8,
    "late_ply": 12,
    "drop": 0.02,
    "judged_depth": 13,
    "break_even_at": 4,
    "bits": [32, 16, 8, 4, 2, 1],
    "actions": 7,
}


def walk_table_bits(entries: int, actions: int) -> int:
    return entries * math.ceil(math.log2(actions))


def exception_bits(exceptions: int, positions: int, actions: int) -> int:
    if exceptions > positions:
        raise ValueError(f"{exceptions} exceptions cannot exceed the tree's {positions} positions")
    if exceptions == 0:
        return 0
    move = math.ceil(math.log2(actions))
    return min(exceptions * (math.ceil(math.log2(positions)) + move), positions + exceptions * move)


def through(by_ply: dict, depth: int) -> int:
    return sum(v for p, v in by_ply.items() if int(p) < depth)


def _integrity(e: dict, spec: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    if e["run"].get("training_fingerprint") != spec["era"]:
        problems.append(f"the nets' era {e['run'].get('training_fingerprint')} is not the registered {spec['era']}")
    if e["table"]["horizon"] != max(spec["depths"]):
        problems.append(f"the table reaches {e['table']['horizon']} plies, not {max(spec['depths'])}")
    if sorted(n["seed"] for n in e["nets"]) != sorted(spec["seeds"]):
        problems.append(f"nets {sorted(n['seed'] for n in e['nets'])} are not the registered {spec['seeds']}")
    for n in e["nets"]:
        want = max(spec["depths"]) if n["seed"] in spec["deep"] else spec["depths"][-2]
        if n["horizon"] != want:
            problems.append(f"seed {n['seed']} reaches {n['horizon']} plies, not {want}")
        if not n["certified"]:
            problems.append(f"seed {n['seed']}'s hybrid did not certify")
        if n["seed"] in spec["deep"] and not n["positions_by_ply"].get(str(spec["late_ply"])):
            problems.append(f"seed {n['seed']} has no positions at ply {spec['late_ply']}")
    return problems


def _rows(e: dict, spec: dict) -> list[dict]:
    rows = []
    for n in sorted(e["nets"], key=lambda n: n["seed"]):
        for depth in (d for d in spec["depths"] if d <= n["horizon"]):
            positions = through(n["positions_by_ply"], depth)
            exceptions = through(n["exceptions_by_ply"], depth)
            entries = through(e["table"]["by_ply"], depth)
            table = walk_table_bits(entries, spec["actions"])
            exc = exception_bits(exceptions, positions, spec["actions"])
            rows.append({"seed": n["seed"], "depth": depth, "positions": positions, "exceptions": exceptions,
                         "table_entries": entries, "table_bits": table, "exception_bits": exc,
                         "break_even": (table - exc) / n["params"],
                         "hybrid_bits": {str(b): n["params"] * b + exc for b in spec["bits"]}})
    return rows


def _share(nets: list, ply: int) -> float:
    return sum(n["exceptions_by_ply"].get(str(ply), 0) for n in nets) / sum(n["positions_by_ply"][str(ply)]
                                                                            for n in nets)


def depth_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if integrity:
        return {"integrity": integrity, "trend": {"verdict": "not_run"}, "break_even": {"verdict": "not_run"}}
    rows = _rows(e, spec)
    deep = [n for n in e["nets"] if n["seed"] in spec["deep"]]
    early, late = _share(deep, spec["early_ply"]), _share(deep, spec["late_ply"])
    trend = "supported" if late <= early - spec["drop"] else "refuted" if late >= early else "inconclusive"
    values = [r["break_even"] for r in rows if r["seed"] in spec["deep"] and r["depth"] == spec["judged_depth"]]
    at = spec["break_even_at"]
    verdict = "supported" if min(values) >= at else "refuted" if max(values) < at else "inconclusive"
    return {"integrity": [], "rows": rows, "trend": {"verdict": trend, "early": early, "late": late},
            "break_even": {"verdict": verdict, "values": values}}
