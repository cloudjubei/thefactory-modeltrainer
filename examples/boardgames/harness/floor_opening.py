"""§C.49 D2 — how much search makes the Connect-4 opening labels right? Pre-registered.

D1 found the solver-free nets wrong in the opening with their own 200-sim search agreeing (h92). At every position
of each saved T10 net's own first-player tree up to `max_ply`, the net's own search is re-run at each registered
budget and the solver judges whether the move it prefers (the argmax of its improved policy — the label training
uses) is optimal; the shallowest plies are also searched at `deep_budget`, reported only.

SUPPORTED when some registered budget prefers an optimal move at >= `support_at` of the positions of EVERY net;
REFUTED when even the largest registered budget, pooled over the nets, prefers an optimal move at fewer than
`refute_below`; INCONCLUSIVE otherwise. Any integrity failure, or unreadable evidence, is NOT_RUN."""
from __future__ import annotations

from harness.floor_plateau import SPEC as PLATEAU_SPEC

SPEC = {"nets": PLATEAU_SPEC["nets"],
        "config": {"budgets": [200, 2000, 20000], "deep_budget": 100000, "deep_plies": 2, "max_ply": 4},
        "era": "ab26151f8373", "measurement_fp": "a3b7d89d2961", "support_at": 0.95, "refute_below": 0.80}


def _problems(evidence, spec: dict) -> list:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("seeds"), list):
        return ["no seeds"]
    cfg = spec["config"]
    problems = []
    if evidence.get("training_fingerprint") != spec["era"]:
        problems.append(f"training era {evidence.get('training_fingerprint')}, registered {spec['era']}")
    if evidence.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if evidence.get("config") != cfg:
        problems.append("config is not the registered measurement")
    rows = {r.get("seed"): r for r in evidence["seeds"] if isinstance(r, dict)}
    if sorted(rows) != sorted(spec["nets"]):
        problems.append(f"seeds are not the registered {sorted(spec['nets'])}")
    for seed, row in rows.items():
        if row.get("net_file_sha256") != spec["nets"].get(seed):
            problems.append(f"seed {seed}: not the saved net registered for it")
        if not row.get("positions"):
            problems.append(f"seed {seed}: no opening positions")
        for p in row.get("positions", []):
            wanted = {str(b) for b in cfg["budgets"]} | ({str(cfg["deep_budget"])} if p["ply"] <= cfg["deep_plies"]
                                                          else set())
            if set(p["preferred"]) != wanted:
                problems.append(f"seed {seed}: a position is not searched at exactly the registered budgets")
            if not p["optimal"] or not 0 <= p["ply"] <= cfg["max_ply"]:
                problems.append(f"seed {seed}: a position has no optimal move or lies beyond ply {cfg['max_ply']}")
    return problems


def _share(positions: list, budget: str) -> float:
    return sum(1 for p in positions if p["preferred"][budget] in p["optimal"]) / len(positions)


def opening_report(evidence, spec: dict = SPEC) -> dict:
    """The pre-registered verdict, or NOT_RUN when the evidence is not the registered measurement."""
    try:
        problems = _problems(evidence, spec)
        if not problems:
            cfg = spec["config"]
            rows = sorted(evidence["seeds"], key=lambda r: r["seed"])
            budgets = [str(b) for b in cfg["budgets"]]
            shares = {b: {r["seed"]: _share(r["positions"], b) for r in rows} for b in budgets}
            everything = [p for r in rows for p in r["positions"]]
            pooled = {b: _share(everything, b) for b in budgets}
            good = [int(b) for b in budgets if min(shares[b].values()) >= spec["support_at"]]
            verdict = ("supported" if good else "refuted" if pooled[budgets[-1]] < spec["refute_below"]
                       else "inconclusive")
            plies = sorted({p["ply"] for p in everything})
            shallow = [p for p in everything if p["ply"] <= cfg["deep_plies"]]
            return {"integrity": [], "verdict": verdict, "first_good_budget": good[0] if good else None,
                    "shares": shares, "pooled": pooled,
                    "by_ply": {b: {str(k): _share([p for p in everything if p["ply"] == k], b) for k in plies}
                               for b in budgets},
                    "deep_share": _share(shallow, str(cfg["deep_budget"])) if shallow else None}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "verdict": "not_run"}
