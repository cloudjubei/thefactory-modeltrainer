"""§C.49 T8 — can the process tell, without a solver, when its net plays perfectly from the start? Pre-registered.

The Connect-4 run must stop without a solver. The candidate signal is the strategy-tree walk's `tree_disagreements`:
how many of the net's raw moves in its own first-player tree its search gave under half the share of its top move.
T8 runs T5's working process (h52: raw-perfect on 10/10 seeds) with the strategy tree switched on, and after every
pass the solver certifies the net's own first-player tree (P-START: every move keeps the game's proven value against
every reply). Iteration i's walk is made by the net after pass i - 1, so its signal is paired with that certificate.

  FALSE STOP  a zero while the net that walked is not certified — the signal would stop on a net that is wrong.
  TIMELY      a seed's first zero comes 0..`latency` iterations after its first certified net.

SUPPORTED when there is no false stop on any seed and at least `support_at` seeds stop in time; REFUTED on any false
stop, or when `refute_at` or fewer seeds stop in time; INCONCLUSIVE between, and when fewer than `certify_at` seeds
ever certify (the stop cannot be calibrated on seeds that never get there). The settled net's certificate is reported
but never paired: no walk follows it. Integrity as harness.arm_judge (same process, same bars on the run itself),
plus the certificates and the walk's readings; any failure, or unreadable evidence, is NOT_RUN."""
from __future__ import annotations

from harness.arm_judge import integrity
from harness.floor_coverage import T5_ARMS

T8_ARM = "augment_sib2_tree"
_CONFIG = {**T5_ARMS["augment_sib2"], "strategy_tree": {"player": 0, "depth": None}}
SPEC = {"arms": {T8_ARM: _CONFIG}, "params": {T8_ARM: _CONFIG["params"]}, "seeds": tuple(range(341, 351)),
        "era": "41605b4ce5d9", "measurement_fp": "2348ed70041f", "iterations": _CONFIG["iterations"],
        "positions": 4520, "latency": 3, "certify_at": 8, "support_at": 8, "refute_at": 5}


def _tree_problems(row: dict, iterations: int) -> list:
    where = f"seed {row.get('seed')}"
    problems = []
    certs = row.get("tree_certification_per_pass")
    if not isinstance(certs, list) or len(certs) != iterations + 1:
        problems.append(f"{where}: not {iterations + 1} tree certificates")
    elif not all(isinstance(c, dict) and isinstance(c.get("certified"), bool) for c in certs):
        problems.append(f"{where}: a tree certificate has no yes/no verdict")
    history = row["history"]
    if history[0].get("tree_walked") != 0:
        problems.append(f"{where}: iteration 0 walked the tree before any relabel")
    for h in history[1:-1]:
        d = h.get("tree_disagreements")
        if not (isinstance(h.get("tree_walked"), int) and h["tree_walked"] > 0):
            problems.append(f"{where}: iteration {h.get('iteration')} did not walk the tree")
        if not isinstance(d, int) or isinstance(d, bool) or d < 0:
            problems.append(f"{where}: iteration {h.get('iteration')} has no disagreement count")
    return problems


def _seed_reading(row: dict, spec: dict) -> dict:
    certs = [c["certified"] for c in row["tree_certification_per_pass"]]
    history, strict = row["history"], row["strict_failures_per_pass"]
    pairs = [(i, history[i]["tree_disagreements"], certs[i - 1]) for i in range(1, spec["iterations"])]
    first_certified = next((i for i, _d, c in pairs if c), None)
    first_zero = next((i for i, d, _c in pairs if d == 0), None)
    latency = None if first_certified is None or first_zero is None else first_zero - first_certified
    stops = [i for i, d, _c in pairs if d == 0]
    return {"seed": row["seed"], "false_stops": [i for i, d, c in pairs if d == 0 and not c],
            "first_certified": first_certified, "first_zero": first_zero, "latency": latency,
            "timely": latency is not None and 0 <= latency <= spec["latency"], "stops": len(stops),
            "stops_strictly_perfect": sum(1 for i in stops if strict[i - 1] == 0),
            "certified_after_settle": certs[-1]}


def _verdict(false_stops: int, certifying: int, timely: int, spec: dict) -> str:
    if false_stops:
        return "refuted"
    if certifying < spec["certify_at"]:
        return "inconclusive"
    return "supported" if timely >= spec["support_at"] else "refuted" if timely <= spec["refute_at"] else "inconclusive"


def stop_report(arm, spec: dict = SPEC) -> dict:
    """The pre-registered verdict on the stop signal from the one arm's evidence, or NOT_RUN when it is not the run."""
    name = next(iter(spec["arms"]))
    try:
        problems = integrity({name: arm}, spec)
        if not problems:
            for row in arm["seeds"]:
                problems.extend(_tree_problems(row, spec["iterations"]))
        if not problems:
            seeds = [_seed_reading(r, spec) for r in sorted(arm["seeds"], key=lambda r: r["seed"])]
            false_stops = sum(len(s["false_stops"]) for s in seeds)
            certifying = sum(1 for s in seeds if s["first_certified"] is not None)
            timely = sum(1 for s in seeds if s["timely"])
            return {"integrity": [], "verdict": _verdict(false_stops, certifying, timely, spec),
                    "false_stops": false_stops, "certifying": certifying, "timely": timely,
                    "stops": sum(s["stops"] for s in seeds),
                    "stops_strictly_perfect": sum(s["stops_strictly_perfect"] for s in seeds), "seeds": seeds}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "verdict": "not_run"}
