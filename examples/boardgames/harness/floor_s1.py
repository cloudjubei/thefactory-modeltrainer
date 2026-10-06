"""§3.6 S1 — discovered steady states at the frontier of the canonical exact Connect-4 table. A fixed random sample of
the table's first-player positions at plies 12 and 14 (ply 12 from evidence/c49_frontier_positions.json.gz, ply 14
derived from it by `next_frontier`) is searched for a
priority map (harness.steady_search, then harness.steady_state.simplify) within a fixed budget; each map found must
pass the solver-free walk (harness.steady_state.verify) and harness.certify. A position is trivial when the rule with
an empty map already wins (win, block, forced moves): it is reported and left out of coverage.

Judged at ply 14 (a pilot found nothing at plies 10-12 within budget, five of seven at ply 14): coverage — found over non-trivial sampled positions — SUPPORTED at >= 30%, REFUTED below 10%; and
compression — 3 bits per first-player position in the state's verified subtree over the map's bits, the median
over found states — SUPPORTED at >= 10x, REFUTED below 1x, INCONCLUSIVE with fewer than five found."""
from __future__ import annotations

import random
import statistics

SPEC = {
    "positions_fp": "a276d6c10259",
    "measurement_fp": "a64ca5bfca25",
    "plies": [12, 14],
    "per_ply": 100,
    "seed": 1,
    "n_levels": 8,
    "level_bits": 3,
    "budget": {"max_constraints": 60_000, "conflicts": 1_000_000, "seconds": 300.0, "cap": 1_000_000, "lines": 64},
    "judged_ply": 14,
    "coverage_support": 0.30,
    "coverage_refute": 0.10,
    "compression_support": 10,
    "compression_refute": 1,
    "min_found": 5,
}


def next_frontier(game, rows: list) -> list[dict]:
    """The table's first-player positions two plies on: from each saved position (with its exact move values), the
    table's move — the first optimal one in action order — then every reply that does not end the game, each
    position once, in the order reached."""
    from games.connect4 import C4State

    seen, out = set(), []
    for row in rows:
        best = max(row["values"].values())
        move = min(int(a) for a, v in row["values"].items() if v == best)
        child = game.step(C4State(tuple(row["board"]), row["to_move"], None, False), move)
        if game.is_terminal(child):
            continue
        for b in game.legal_actions(child):
            s = game.step(child, b)
            key = game.state_key(s)
            if not game.is_terminal(s) and key not in seen:
                seen.add(key)
                out.append({"board": list(s.board), "to_move": s.to_move})
    return out


def sample_indices(total: int, per_ply: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(total), per_ply))


def pilot_indices(total: int, per_ply: int, seed: int, n: int) -> list[int]:
    taken = set(sample_indices(total, per_ply, seed))
    return sorted(random.Random(seed + 1).sample([i for i in range(total) if i not in taken], n))


def _integrity(e: dict, spec: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    if e.get("positions_fingerprint") != spec["positions_fp"]:
        problems.append(f"positions {e.get('positions_fingerprint')} are not the registered {spec['positions_fp']}")
    if e.get("config") != {k: spec[k] for k in ("n_levels", "level_bits", "budget")}:
        problems.append("the search ran with another language or budget than registered")
    for ply in spec["plies"]:
        n = sum(1 for r in e["results"] if r["ply"] == ply)
        if n != spec["per_ply"]:
            problems.append(f"ply {ply}: {n} positions, not {spec['per_ply']}")
    if any(r["status"] == "found" and not (r["verified"] and r["certified"]) for r in e["results"]):
        problems.append("a found map failed an independent check")
    return problems


def _ply_summary(rows: list) -> dict:
    found = [r for r in rows if r["status"] == "found"]
    live = [r for r in rows if not r["trivial"]]
    return {"sampled": len(rows), "trivial": len(rows) - len(live), "found": len(found),
            "impossible": sum(1 for r in rows if r["status"] == "impossible"),
            "budget": sum(1 for r in rows if r["status"] == "budget"),
            "coverage": len(found) / len(live) if live else 0.0,
            "median_bits": statistics.median(r["bits"] for r in found) if found else None,
            "median_own_positions": statistics.median(r["own_positions"] for r in found) if found else None}


def s1_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if integrity:
        return {"integrity": integrity, "coverage": {"verdict": "not_run"}, "compression": {"verdict": "not_run"}}
    plies = {str(p): _ply_summary([r for r in e["results"] if r["ply"] == p]) for p in spec["plies"]}
    judged = plies[str(spec["judged_ply"])]
    c = judged["coverage"]
    coverage = ("supported" if c >= spec["coverage_support"] else "refuted" if c < spec["coverage_refute"]
                else "inconclusive")
    values = [3 * r["own_positions"] / r["bits"] for r in e["results"]
              if r["ply"] == spec["judged_ply"] and r["status"] == "found"]
    median = statistics.median(values) if values else None
    if len(values) < spec["min_found"]:
        compression = "inconclusive"
    else:
        compression = ("supported" if median >= spec["compression_support"]
                       else "refuted" if median < spec["compression_refute"] else "inconclusive")
    return {"integrity": [], "plies": plies, "coverage": {"verdict": coverage, "value": c},
            "compression": {"verdict": compression, "median": median, "values": values}}
