"""§C.49 T20 — is the Connect-4 process still learning at iteration 20? Pre-registered.

Five fixes (h111-h126) were judged on 20-iteration pilots, and every arm ends playing the opening at about 71%. If
the process is still improving at iteration 20, the pilots were too short to show an opening effect at all; if it has
flattened, the opening is out of this process's reach as it stands. T20 runs the base pilot recipe for 60 iterations
on three fresh seeds and scores the net on the fixed set of exactly valued positions (harness.fixed_set) after every
training pass (an observation only), one run file per seed.

Per seed the share at the `early` passes (around the pilots' length) and at the `late` passes is averaged, and the
pooled late-minus-early gain is judged twice — at the `opening` plies and over all positions: SUPPORTED (still
learning) at a gain of at least `support_at`, REFUTED (flat) at or below `refute_at`, INCONCLUSIVE between. Evidence
that is not the registered run is NOT_RUN. Pass p is the reading after the p-th training pass (curve[p - 1])."""
from __future__ import annotations

from statistics import mean

from harness.floor_c4_value_signal import SPEC as T14_SPEC

SPEC = {"config": {**T14_SPEC["arms"]["base"], "iterations": 60}, "seeds": (471, 472, 473), "era": "13284aea9fa6",
        "measurement_fp": "68a23ee868fa", "early": [16, 17, 18, 19, 20], "late": [56, 57, 58, 59, 60],
        "opening": ["0", "2", "4"], "support_at": 0.03, "refute_at": 0.01}


def _problems(runs: list, spec: dict) -> list:
    problems = []
    for run in runs:
        if run.get("training_fingerprint") != spec["era"]:
            problems.append(f"training era {run.get('training_fingerprint')}, registered {spec['era']}")
        if run.get("measurement_fingerprint") != spec["measurement_fp"]:
            problems.append(f"measurement code {run.get('measurement_fingerprint')}, registered "
                            f"{spec['measurement_fp']}")
        config = {k: v for k, v in run.get("config", {}).items() if k not in ("seeds", "certify_depth")}
        if config != spec["config"] or sorted(r["seed"] for r in run["seeds"]) != sorted(run["config"]["seeds"]):
            problems.append("config is not the registered recipe")
        for row in run["seeds"]:
            if len(row.get("curve") or []) != spec["config"]["iterations"] + 1:
                problems.append(f"seed {row['seed']}: not one reading per training pass")
    if sorted(r["seed"] for run in runs for r in run["seeds"]) != list(spec["seeds"]):
        problems.append(f"seeds are not the registered {list(spec['seeds'])}")
    return problems


def _share(point: dict, plies) -> float:
    if plies is None:
        return point["optimal"] / point["positions"]
    rows = [point["by_ply"][p] for p in plies]
    return sum(r["optimal"] for r in rows) / sum(r["positions"] for r in rows)


def _reading(rows: list, spec: dict, plies) -> dict:
    seeds = {}
    for row in sorted(rows, key=lambda r: r["seed"]):
        at = [_share(point, plies) for point in row["curve"]]
        seeds[str(row["seed"])] = {"early": mean(at[p - 1] for p in spec["early"]),
                                   "late": mean(at[p - 1] for p in spec["late"])}
    gain = round(mean(s["late"] - s["early"] for s in seeds.values()), 9)
    verdict = ("supported" if gain >= spec["support_at"] else "refuted" if gain <= spec["refute_at"]
               else "inconclusive")
    return {"verdict": verdict, "gain": gain, "seeds": seeds}


def curve_report(runs: list, spec: dict = SPEC) -> dict:
    """Both pre-registered readings from the seeds' run files, or NOT_RUN for both."""
    try:
        problems = _problems(runs, spec)
        if not problems:
            rows = [r for run in runs for r in run["seeds"]]
            return {"integrity": [], "opening": _reading(rows, spec, spec["opening"]),
                    "overall": _reading(rows, spec, None)}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "opening": {"verdict": "not_run"}, "overall": {"verdict": "not_run"}}
