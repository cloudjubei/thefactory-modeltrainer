"""Direct tests for harness/floor_c4_value.py — T12: T10's solver-free Connect-4 run with the value-aware stop,
judged by T10's own judge (harness.floor_c4.c4_report) on its own registered recipe."""
from __future__ import annotations

from harness.floor_c4 import SPEC as T10_SPEC, c4_report
from harness.floor_c4_value import SPEC


def _row(seed, stopped=True, certified=True):
    cfg = SPEC["config"]
    history = [{"iteration": 1, "tree_walked": 0, "tree_disagreements": 0, "siblings": 0}]
    history += [{"iteration": i, "tree_walked": 40, "tree_disagreements": 3, "siblings": 0} for i in (2, 3)]
    if stopped:
        history.append({"iteration": 4, "stopped": True, "tree_walked": 40, "tree_disagreements": 0})
        games = [cfg["selfplay"]] * 3
    else:
        history += [{"iteration": i, "tree_walked": 40, "tree_disagreements": 2, "siblings": 0}
                    for i in range(4, cfg["iterations"] + 1)]
        history.append({"iteration": "settle"})
        games = [cfg["selfplay"]] * cfg["iterations"] + [0]
    return {"seed": seed, "stopped": stopped, "params": SPEC["params"], "net": {"file_sha256": "f"},
            "games_per_pass": games, "history": history,
            "certificate": {"certified": certified, "net_file_sha256": "f", "horizon": SPEC["certify_depth"]}}


def _evidence(rows=None, config=None):
    rows = rows or {}
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": {**(config or SPEC["config"]), "seeds": list(SPEC["seeds"]),
                       "certify_depth": SPEC["certify_depth"]},
            "seeds": [_row(s, **rows.get(s, {})) for s in SPEC["seeds"]]}


def test_the_recipe_is_t10_s_with_the_value_delta_and_fresh_seeds():
    assert SPEC["config"] == {**T10_SPEC["config"], "stop_value_delta": 0.1}
    assert not set(SPEC["seeds"]) & set(T10_SPEC["seeds"]) and len(SPEC["seeds"]) == 10
    assert {k: v for k, v in SPEC.items() if k not in ("config", "seeds", "era", "measurement_fp")} == \
        {k: v for k, v in T10_SPEC.items() if k not in ("config", "seeds", "era", "measurement_fp")}


def test_stopped_and_certified_seeds_support_and_false_stops_are_reported():
    r = c4_report(_evidence({381: {"certified": False}}), SPEC)
    assert r["verdict"] == "supported" and r["successes"] == 9 and r["descriptives"]["false_stops"] == 1


def test_a_run_on_t10_s_recipe_is_not_t12():
    assert c4_report(_evidence(config=T10_SPEC["config"]), SPEC)["verdict"] == "not_run"
