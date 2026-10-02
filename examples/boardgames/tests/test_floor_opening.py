"""Direct tests for harness/floor_opening.py — D2: how much search makes the opening labels right? At every opening
position of each saved net's own first-player tree, the solver judges the move the net's own search prefers at each
registered budget."""
from __future__ import annotations

import pytest

from harness.floor_opening import opening_report

SPEC = {"nets": {1: "a" * 64, 2: "b" * 64},
        "config": {"budgets": [200, 2000], "deep_budget": 9000, "deep_plies": 2, "max_ply": 4},
        "era": "e" * 12, "measurement_fp": "m" * 12, "support_at": 0.95, "refute_below": 0.80}


def _pos(ply, right=(True, True), deep=True):
    preferred = {"200": 0 if right[0] else 1, "2000": 0 if right[1] else 1}
    if ply <= SPEC["config"]["deep_plies"]:
        preferred["9000"] = 0 if deep else 1
    return {"ply": ply, "board": [0] * 42, "to_move": 0, "optimal": [0], "preferred": preferred}


def _row(seed, wrong_200=0, wrong_2000=0):
    positions = [_pos(0)] + [_pos(2) for _ in range(4)] + [_pos(4) for _ in range(15)]
    for i in range(wrong_200):
        positions[i + 1]["preferred"]["200"] = 1
    for i in range(wrong_2000):
        positions[i + 1]["preferred"]["2000"] = 1
    return {"seed": seed, "net_file_sha256": SPEC["nets"][seed], "positions": positions}


def _evidence(rows=None):
    rows = rows or {}
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": {**SPEC["config"], "budgets": list(SPEC["config"]["budgets"])},
            "seeds": [_row(s, **rows.get(s, {})) for s in SPEC["nets"]]}


def test_a_budget_right_at_95_percent_of_every_net_s_opening_positions_supports():
    r = opening_report(_evidence({1: {"wrong_200": 5}, 2: {"wrong_200": 3}}), SPEC)
    assert r["integrity"] == [] and r["verdict"] == "supported" and r["first_good_budget"] == 2000
    assert r["shares"]["200"] == {1: 0.75, 2: 0.85} and r["shares"]["2000"] == {1: 1.0, 2: 1.0}


def test_exactly_95_percent_on_every_net_is_enough():
    r = opening_report(_evidence({1: {"wrong_200": 1, "wrong_2000": 1}, 2: {"wrong_200": 1}}), SPEC)
    assert r["shares"]["200"] == {1: 0.95, 2: 0.95} and r["verdict"] == "supported" and r["first_good_budget"] == 200


def test_the_refutation_reads_the_largest_budget_and_exactly_80_percent_is_not_below_it():
    weak_small = opening_report(_evidence({1: {"wrong_200": 6, "wrong_2000": 2}, 2: {"wrong_200": 3}}), SPEC)
    assert weak_small["pooled"]["200"] < 0.8 and weak_small["verdict"] == "inconclusive"
    at_bar = opening_report(_evidence({1: {"wrong_200": 4, "wrong_2000": 4}, 2: {"wrong_200": 4, "wrong_2000": 4}}),
                            SPEC)
    assert at_bar["pooled"]["2000"] == pytest.approx(0.8) and at_bar["verdict"] == "inconclusive"


def test_a_net_without_opening_positions_is_named():
    e = _evidence()
    e["seeds"][1]["positions"] = []
    assert any("no opening positions" in p for p in opening_report(e, SPEC)["integrity"])


def test_one_net_below_the_bar_at_every_budget_is_not_supported():
    r = opening_report(_evidence({1: {"wrong_200": 2, "wrong_2000": 2}}), SPEC)
    assert r["verdict"] == "inconclusive" and r["first_good_budget"] is None


def test_the_largest_budget_still_wrong_too_often_refutes():
    r = opening_report(_evidence({1: {"wrong_200": 6, "wrong_2000": 6}, 2: {"wrong_200": 3, "wrong_2000": 3}}), SPEC)
    assert r["pooled"]["2000"] == pytest.approx(31 / 40) and r["verdict"] == "refuted"


def test_the_deep_budget_is_reported_only_for_the_shallowest_plies():
    e = _evidence()
    e["seeds"][0]["positions"][1]["preferred"]["9000"] = 1
    r = opening_report(e, SPEC)
    assert r["deep_share"] == pytest.approx(9 / 10) and r["verdict"] == "supported"


def test_shares_are_also_broken_out_by_ply():
    r = opening_report(_evidence({1: {"wrong_200": 4}}), SPEC)
    assert r["by_ply"]["200"]["2"] == pytest.approx(4 / 8) and r["by_ply"]["200"]["4"] == 1.0


def _set(path, value):
    def apply(e):
        target = e
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("training_fingerprint",), "0" * 12),
    "measurement": _set(("measurement_fingerprint",), "0" * 12),
    "config": _set(("config", "budgets"), [200]),
    "another net": _set(("seeds", 0, "net_file_sha256"), "z" * 64),
    "missing seed": lambda e: e["seeds"].pop(),
    "no positions": _set(("seeds", 1, "positions"), []),
    "an unregistered budget": _set(("seeds", 0, "positions", 3, "preferred", "500"), 0),
    "deep budget missing": lambda e: e["seeds"][0]["positions"][0]["preferred"].pop("9000"),
    "no optimal move": _set(("seeds", 0, "positions", 5, "optimal"), []),
    "too deep": _set(("seeds", 1, "positions", 6, "ply"), 6),
}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_evidence_that_is_not_the_registered_run_is_not_run(breakage):
    e = _evidence()
    BREAKAGES[breakage](e)
    r = opening_report(e, SPEC)
    assert r["integrity"] and r["verdict"] == "not_run"


@pytest.mark.parametrize("e", [None, {}, {"seeds": "x"}, []])
def test_missing_or_unreadable_evidence_is_not_run(e):
    assert opening_report(e, SPEC)["verdict"] == "not_run"
