"""§C.49 T11 — the VALUE-AWARE stop signal, recalibrated on tic-tac-toe under a deliberately weakened search.
Pre-registered.

On Connect-4 the share rule never reads zero: 59% of the disagreements there are between two optimal moves (h90), so
the stop now also counts a raw move as agreeing when the search's Q for it is within `stop_value_delta` of the Q of
the label's top move. That makes stopping easier, so it must be shown not to stop on a wrong net where the search is
weak — the Connect-4 situation, where more search did not fix the opening (h96). T11 runs T9's process with the
relabel search cut from 200 to 50 simulations and the value-aware reading recorded at every iteration (the run does
not stop on it), and judges it exactly as floor_tree judges T9's stop: no false stop on any seed, and the first zero
within `latency` of the first certified net on enough seeds.

The share rule's reading is recorded beside it. Values only add agreements, so the share rule never reads zero where
the value rule does not: only when it would have stopped is reported (not judged). A walked iteration without the
share reading, or with a value reading above it, makes the run NOT_RUN, as does a recipe without the delta."""
from __future__ import annotations

from harness.floor_stop import _seed_reading
from harness.floor_tree import _CONFIG as T9_CONFIG, tree_report

T11_ARM = "tree_value_stop_weak"
_CONFIG = {**T9_CONFIG, "reanalyze_sims": 50, "stop_value_delta": 0.1}
SPEC = {"arms": {T11_ARM: _CONFIG}, "params": {T11_ARM: _CONFIG["params"]}, "seeds": tuple(range(371, 381)),
        "era": "0c8fa3e34fdc", "measurement_fp": "ee230228a735", "iterations": _CONFIG["iterations"],
        "positions": 4520, "latency": 3, "certify_at": 8, "support_at": 8, "refute_at": 5}


def _share_problems(arm) -> list:
    problems = []
    for row in arm["seeds"]:
        for h in row["history"][1:-1]:
            share, value = h.get("tree_disagreements_share"), h["tree_disagreements"]
            if not isinstance(share, int) or isinstance(share, bool) or share < value:
                problems.append(f"seed {row['seed']}: iteration {h.get('iteration')} has no share reading at or "
                                f"above its value reading")
    return problems


def _share_row(row: dict) -> dict:
    history = [{**h, "tree_disagreements": h.get("tree_disagreements_share")} for h in row["history"]]
    return {**row, "history": history}


def value_stop_report(arm, spec: dict = SPEC) -> dict:
    """The pre-registered stop verdict on the one arm's evidence, the share rule's readings beside it, or NOT_RUN."""
    if "stop_value_delta" not in spec["arms"][next(iter(spec["arms"]))]:
        return {"integrity": ["the registered recipe has no stop_value_delta — this is not a value-stop run"],
                "stop": {"verdict": "not_run"}}
    report = tree_report(arm, spec)
    if report["integrity"]:
        return {"integrity": report["integrity"], "stop": {"verdict": "not_run"}}
    problems = _share_problems(arm)
    if problems:
        return {"integrity": problems, "stop": {"verdict": "not_run"}}
    rows = sorted(arm["seeds"], key=lambda r: r["seed"])
    share = [_seed_reading(_share_row(r), spec) for r in rows]
    return {"integrity": [], "stop": report["stop"], "pstart": report["pstart"],
            "share": {"timely": sum(1 for s in share if s["timely"]), "first_zero": [s["first_zero"] for s in share]},
            "descriptives": report["descriptives"]}
