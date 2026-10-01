"""Direct tests for harness/floor_tree.py — T9: the strategy tree INSTEAD of siblings, judged on tic-tac-toe by two
pre-registered readings of one run: does the settled net play perfectly from the start as the first player
(P-START), and is the stop signal still safe and timely."""
from __future__ import annotations

import pytest

from harness.floor_tree import tree_report

ARM = "a"
SPEC = {"arms": {ARM: {"selfplay": 4, "settle_epochs": 7, "strategy_tree": {"player": 0}}}, "params": {ARM: 100},
        "seeds": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "era": "e" * 12, "measurement_fp": "m" * 12, "iterations": 6,
        "positions": 50, "latency": 2, "certify_at": 8, "support_at": 8, "refute_at": 5}


def _row(seed, disagreements=(4, 2, 1, 0, 0), certified=(False, False, True, True, True, True, True), final=0):
    cfg, n = SPEC["arms"][ARM], SPEC["iterations"]
    history = [{"iteration": 1, "merged": 3, "siblings": 0, "tree_walked": 0, "tree_positions": 0,
                "tree_disagreements": 0}]
    history += [{"iteration": i + 1, "merged": 3, "siblings": 0, "tree_walked": 9, "tree_positions": 2,
                 "tree_disagreements": d} for i, d in enumerate(disagreements, start=1)]
    history.append({"iteration": "settle", "epochs": cfg["settle_epochs"]})
    return {"seed": seed, "params": SPEC["params"][ARM], "positions": SPEC["positions"],
            "strict_failures_per_pass": [3] * n + [final], "final_failing_positions": list(range(final)),
            "games_per_pass": [cfg["selfplay"]] * n + [0], "history": history, "train_seconds": 10.0,
            "tree_certification_per_pass": [{"certified": c, "failures": 0 if c else 2} for c in certified]}


def _arm(rows=None):
    rows = rows or {}
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": {**SPEC["arms"][ARM], "seeds": list(SPEC["seeds"])},
            "seeds": [_row(s, **rows.get(s, {})) for s in SPEC["seeds"]]}


def test_a_run_certified_after_the_settle_with_a_timely_safe_stop_on_every_seed_supports_both_readings():
    r = tree_report(_arm(), SPEC)
    assert r["integrity"] == [] and r["pstart"]["verdict"] == "supported" and r["pstart"]["certified"] == 10
    assert r["stop"]["verdict"] == "supported" and r["stop"]["false_stops"] == 0 and r["stop"]["timely"] == 10
    assert r["descriptives"]["raw_perfect"] == 10


@pytest.mark.parametrize("uncertified,verdict", [(0, "supported"), (2, "supported"), (3, "inconclusive"),
                                                 (4, "inconclusive"), (5, "refuted"), (10, "refuted")])
def test_the_start_reading_counts_seeds_whose_settled_net_is_certified(uncertified, verdict):
    lost = {"certified": (False, False, True, True, True, True, False), "disagreements": (4, 2, 1, 0, 0)}
    r = tree_report(_arm({s: lost for s in SPEC["seeds"][:uncertified]}), SPEC)
    assert r["pstart"]["certified"] == 10 - uncertified and r["pstart"]["verdict"] == verdict


def test_the_stop_reading_is_floor_stop_s_on_the_same_run():
    early = {"disagreements": (4, 0, 0, 0, 0), "certified": (False, False, True, True, True, True, True)}
    r = tree_report(_arm({1: early}), SPEC)
    assert r["stop"]["false_stops"] == 1 and r["stop"]["verdict"] == "refuted"
    assert r["pstart"]["verdict"] == "supported"


def test_all_position_perfection_is_reported_not_judged():
    r = tree_report(_arm({s: {"final": 2} for s in SPEC["seeds"][:6]}), SPEC)
    assert r["descriptives"]["raw_perfect"] == 4 and r["pstart"]["verdict"] == "supported"


def _set(path, value):
    def apply(arm):
        target = arm
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("training_fingerprint",), "0" * 12),
    "measurement": _set(("measurement_fingerprint",), "0" * 12),
    "config": _set(("config", "strategy_tree"), None),
    "seeds": _set(("config", "seeds"), [1, 2]),
    "seed rows": _set(("seeds", 0, "seed"), 99),
    "params": _set(("seeds", 1, "params"), 7),
    "positions": _set(("seeds", 2, "positions"), 49),
    "passes": _set(("seeds", 3, "strict_failures_per_pass"), [3] * 7 + [0]),
    "games": _set(("seeds", 4, "games_per_pass"), [4] * 7),
    "no settle": _set(("seeds", 5, "history", 6, "iteration"), 7),
    "settle epochs": _set(("seeds", 5, "history", 6, "epochs"), 30),
    "siblings came back": _set(("seeds", 6, "history", 2, "siblings"), 5),
    "no unique buffer": lambda arm: arm["seeds"][7]["history"][1].pop("merged"),
    "certificates": _set(("seeds", 8, "tree_certification_per_pass"), [{"certified": True}] * 6),
    "no walk": _set(("seeds", 9, "history", 2, "tree_walked"), 0),
}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_evidence_that_is_not_the_registered_run_is_not_run(breakage):
    arm = _arm()
    BREAKAGES[breakage](arm)
    r = tree_report(arm, SPEC)
    assert r["integrity"] and r["pstart"]["verdict"] == "not_run" and r["stop"]["verdict"] == "not_run"


@pytest.mark.parametrize("arm", [None, {}, {"seeds": "x"}, []])
def test_missing_or_unreadable_evidence_is_not_run(arm):
    r = tree_report(arm, SPEC)
    assert r["pstart"]["verdict"] == "not_run" and r["stop"]["verdict"] == "not_run"
