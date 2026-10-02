"""Direct tests for harness/floor_value_stop.py — T11: the value-aware stop signal recalibrated on tic-tac-toe under a
weakened search. Its verdict is floor_tree's stop reading of the run; on top it needs the share reading recorded
beside the value reading at every walked iteration, and reports when the share rule alone would have stopped — it
never reads zero where the value rule does not, so only its timeliness is worth comparing."""
from __future__ import annotations

import pytest

from harness.floor_value_stop import value_stop_report

ARM = "a"
SPEC = {"arms": {ARM: {"selfplay": 4, "settle_epochs": 7, "strategy_tree": {"player": 0}, "stop_value_delta": 0.1}},
        "params": {ARM: 100}, "seeds": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "era": "e" * 12, "measurement_fp": "m" * 12,
        "iterations": 6, "positions": 50, "latency": 2, "certify_at": 8, "support_at": 8, "refute_at": 5}
CERTIFIED = (False, False, True, True, True, True, True)


def _row(seed, value=(4, 2, 1, 0, 0), share=(5, 3, 2, 1, 0), certified=CERTIFIED):
    cfg, n = SPEC["arms"][ARM], SPEC["iterations"]
    history = [{"iteration": 1, "merged": 3, "siblings": 0, "tree_walked": 0, "tree_positions": 0,
                "tree_disagreements": 0, "tree_disagreements_share": 0}]
    history += [{"iteration": i + 1, "merged": 3, "siblings": 0, "tree_walked": 9, "tree_positions": 2,
                 "tree_disagreements": v, "tree_disagreements_share": s}
                for i, (v, s) in enumerate(zip(value, share), start=1)]
    history.append({"iteration": "settle", "epochs": cfg["settle_epochs"]})
    return {"seed": seed, "params": SPEC["params"][ARM], "positions": SPEC["positions"],
            "strict_failures_per_pass": [3] * n + [0], "final_failing_positions": [],
            "games_per_pass": [cfg["selfplay"]] * n + [0], "history": history, "train_seconds": 10.0,
            "tree_certification_per_pass": [{"certified": c, "failures": 0 if c else 2} for c in certified]}


def _arm(rows=None):
    rows = rows or {}
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": {**SPEC["arms"][ARM], "seeds": list(SPEC["seeds"])},
            "seeds": [_row(s, **rows.get(s, {})) for s in SPEC["seeds"]]}


def test_a_safe_timely_value_stop_on_every_seed_is_supported_and_the_share_rule_is_reported_beside_it():
    r = value_stop_report(_arm(), SPEC)
    assert r["integrity"] == [] and r["stop"]["verdict"] == "supported" and r["stop"]["timely"] == 10
    assert r["share"] == {"timely": 10, "first_zero": [5] * 10}


def test_a_value_zero_before_the_net_is_certified_refutes_the_stop():
    r = value_stop_report(_arm({3: {"value": (4, 0, 1, 0, 0), "share": (5, 1, 2, 1, 0)}}), SPEC)
    assert r["stop"]["verdict"] == "refuted" and r["stop"]["false_stops"] == 1


def test_the_share_rule_s_timeliness_is_reported_beside_the_value_rule_s_but_does_not_judge():
    r = value_stop_report(_arm({s: {"share": (5, 3, 2, 1, 1)} for s in SPEC["seeds"][:6]}), SPEC)
    assert r["share"] == {"timely": 4, "first_zero": [None] * 6 + [5] * 4} and r["stop"]["verdict"] == "supported"


@pytest.mark.parametrize("value,share,problem", [
    ((4, 2, 1, 0, 0), (5, 3, None, 1, 0), "share reading"),
    ((4, 2, 1, 0, 0), (5, 1, 2, 1, 0), "share reading"),
])
def test_a_missing_or_impossible_share_reading_makes_the_run_not_run(value, share, problem):
    r = value_stop_report(_arm({5: {"value": value, "share": share}}), SPEC)
    assert r["stop"]["verdict"] == "not_run" and any(problem in p for p in r["integrity"])


def test_floor_tree_s_integrity_still_applies():
    arm = _arm()
    arm["training_fingerprint"] = "0" * 12
    assert value_stop_report(arm, SPEC)["stop"]["verdict"] == "not_run"


def test_a_run_without_the_value_delta_in_its_recipe_is_not_the_run():
    spec = {**SPEC, "arms": {ARM: {k: v for k, v in SPEC["arms"][ARM].items() if k != "stop_value_delta"}}}
    r = value_stop_report(_arm(), spec)
    assert r["stop"]["verdict"] == "not_run" and any("stop_value_delta" in p for p in r["integrity"])
