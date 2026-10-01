"""Direct tests for harness/floor_stop.py — the pre-registered calibration of the solver-free stop signal: each
relabelled iteration's count of raw moves the search disagrees with is paired with the solver's certificate of the
net that walked, a zero while that net is not certified is a FALSE STOP, and a seed stops in time when its first
zero comes within the registered latency of its first certified net."""
from __future__ import annotations

import pytest

from harness.floor_stop import stop_report

ARM = "a"
SPEC = {"arms": {ARM: {"selfplay": 4, "settle_epochs": 7, "sibling_depth": 2, "strategy_tree": {"player": 0}}},
        "params": {ARM: 100}, "seeds": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "era": "e" * 12, "measurement_fp": "m" * 12,
        "iterations": 6, "positions": 50, "latency": 2, "certify_at": 8, "support_at": 8, "refute_at": 5}


def _row(seed, disagreements, certified):
    """`disagreements` for iterations 1..5 (iteration 0 walks nothing); `certified` for passes 0..6 (the last is the
    settle)."""
    cfg, n = SPEC["arms"][ARM], SPEC["iterations"]
    history = [{"iteration": 1, "merged": 3, "sibling_rings": [], "tree_walked": 0, "tree_positions": 0,
                "tree_disagreements": 0}]
    history += [{"iteration": i + 1, "merged": 3, "sibling_rings": [2] * cfg["sibling_depth"], "tree_walked": 9,
                 "tree_positions": 2, "tree_disagreements": d} for i, d in enumerate(disagreements, start=1)]
    history.append({"iteration": "settle", "epochs": cfg["settle_epochs"]})
    return {"seed": seed, "params": SPEC["params"][ARM], "positions": SPEC["positions"],
            "strict_failures_per_pass": [3] * n + [0], "final_failing_positions": [],
            "games_per_pass": [cfg["selfplay"]] * n + [0], "history": history,
            "tree_certification_per_pass": [{"certified": c, "failures": 0 if c else 2} for c in certified]}


TIMELY = ([4, 2, 1, 0, 0], [False, False, True, True, True, True, True])


def _arm(rows=None):
    rows = rows or {}
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": {**SPEC["arms"][ARM], "seeds": list(SPEC["seeds"])},
            "seeds": [_row(s, *rows.get(s, TIMELY)) for s in SPEC["seeds"]]}


def test_every_seed_stopping_within_the_latency_of_its_first_certified_net_with_no_false_stop_is_supported():
    r = stop_report(_arm(), SPEC)
    assert r["integrity"] == [] and r["verdict"] == "supported"
    assert r["false_stops"] == 0 and r["certifying"] == 10 and r["timely"] == 10
    seed = r["seeds"][0]
    assert seed["first_certified"] == 3 and seed["first_zero"] == 4 and seed["latency"] == 1


def test_iteration_i_s_signal_is_paired_with_the_net_after_pass_i_minus_one():
    only_later = ([4, 2, 0, 0, 0], [False, False, False, True, True, True, True])
    r = stop_report(_arm({1: only_later}), SPEC)
    assert r["seeds"][0]["false_stops"] == [3] and r["verdict"] == "refuted"
    assert r["seeds"][0]["latency"] == -1 and not r["seeds"][0]["timely"] and r["timely"] == 9


def test_a_single_false_stop_on_any_seed_refutes_even_after_certification():
    relapse = ([4, 0, 0, 0, 0], [False, True, True, False, True, True, True])
    r = stop_report(_arm({5: relapse}), SPEC)
    assert r["false_stops"] == 1 and r["seeds"][4]["false_stops"] == [4] and r["verdict"] == "refuted"


@pytest.mark.parametrize("late,verdict", [(0, "supported"), (2, "supported"), (3, "inconclusive"),
                                          (4, "inconclusive"), (5, "refuted"), (10, "refuted")])
def test_seeds_stopping_later_than_the_latency_count_against_the_bars(late, verdict):
    slow = ([4, 2, 1, 1, 0], [False, True, True, True, True, True, True])
    r = stop_report(_arm({s: slow for s in SPEC["seeds"][:late]}), SPEC)
    assert r["false_stops"] == 0 and r["timely"] == 10 - late and r["verdict"] == verdict
    if late:
        assert r["seeds"][0]["latency"] == 3


def test_a_first_zero_exactly_the_latency_after_certification_is_in_time():
    at_bar = ([4, 2, 1, 0, 0], [False, True, True, True, True, True, True])
    r = stop_report(_arm({s: at_bar for s in SPEC["seeds"]}), SPEC)
    assert r["seeds"][0]["latency"] == SPEC["latency"] and r["timely"] == 10 and r["verdict"] == "supported"


def test_a_certified_seed_that_never_reaches_zero_is_not_timely():
    never = ([4, 2, 1, 1, 1], [False, True, True, True, True, True, True])
    r = stop_report(_arm({s: never for s in SPEC["seeds"][:3]}), SPEC)
    assert r["seeds"][0]["first_zero"] is None and r["seeds"][0]["latency"] is None
    assert r["timely"] == 7 and r["verdict"] == "inconclusive"


@pytest.mark.parametrize("uncertified,verdict", [(2, "supported"), (3, "inconclusive"), (10, "inconclusive")])
def test_too_few_certified_seeds_leave_the_stop_uncalibrated(uncertified, verdict):
    stuck = ([4, 3, 3, 2, 2], [False] * 7)
    r = stop_report(_arm({s: stuck for s in SPEC["seeds"][:uncertified]}), SPEC)
    assert r["certifying"] == 10 - uncertified and r["verdict"] == verdict


def test_the_settled_net_s_certificate_is_reported_but_never_paired():
    settle_only = ([4, 2, 1, 1, 1], [False] * 6 + [True])
    r = stop_report(_arm({s: settle_only for s in SPEC["seeds"]}), SPEC)
    assert r["certifying"] == 0 and r["false_stops"] == 0 and r["verdict"] == "inconclusive"
    assert r["seeds"][0]["certified_after_settle"] is True


def test_the_report_says_how_often_a_stop_landed_on_a_net_perfect_at_every_position():
    arm = _arm()
    arm["seeds"][0]["strict_failures_per_pass"] = [3, 3, 3, 0, 3, 3, 0]
    r = stop_report(arm, SPEC)
    assert r["stops"] == 20 and r["stops_strictly_perfect"] == 1


def _set(path, value):
    def apply(arm):
        target = arm
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("training_fingerprint",), "0" * 12),
    "config": _set(("config", "strategy_tree"), None),
    "base checks": _set(("seeds", 1, "games_per_pass"), [4] * 7),
    "certificates": _set(("seeds", 2, "tree_certification_per_pass"), [{"certified": True}] * 6),
    "certificate shape": _set(("seeds", 3, "tree_certification_per_pass", 0), {"certified": 1}),
    "walk on iteration 0": _set(("seeds", 4, "history", 0, "tree_walked"), 3),
    "no walk": _set(("seeds", 5, "history", 2, "tree_walked"), 0),
    "no reading": lambda arm: arm["seeds"][6]["history"][3].pop("tree_disagreements"),
    "reading not a count": _set(("seeds", 7, "history", 1, "tree_disagreements"), 0.5),
}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_evidence_that_is_not_the_registered_run_is_not_run(breakage):
    arm = _arm()
    BREAKAGES[breakage](arm)
    r = stop_report(arm, SPEC)
    assert r["integrity"] and r["verdict"] == "not_run"


@pytest.mark.parametrize("arm", [None, {}, {"seeds": "x"}, []])
def test_missing_or_unreadable_evidence_is_not_run(arm):
    assert stop_report(arm, SPEC)["verdict"] == "not_run"
