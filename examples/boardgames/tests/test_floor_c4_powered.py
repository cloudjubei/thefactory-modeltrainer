"""Direct tests for harness/floor_c4_powered.py — T17: the powered, pre-registered comparison of three value-signal
fixes against base on Connect-4, every net scored on the same fixed position set, each treatment judged by an exact
one-sided sign-flip permutation test on its seed-matched differences, with Holm's correction across treatments."""
from __future__ import annotations

import pytest

from harness.floor_c4_powered import SPEC, holm, permutation_p, powered_report

TEST_SPEC = {"arms": {"base": {"a": 0}, "x": {"a": 1}, "y": {"a": 2}}, "treatments": ("x", "y"),
             "seeds": (1, 2, 3, 4, 5, 6, 7), "prefix": "P", "era": "e" * 12, "measurement_fp": "m" * 12,
             "plies": [0, 2], "positions": 1000, "certify_depth": 6, "alpha": 0.05}


def test_the_permutation_p_counts_sign_flips_at_least_as_favourable_as_the_observed_differences():
    assert permutation_p([0.01] * 7) == pytest.approx(1 / 128)
    assert permutation_p([0.01] * 6 + [-0.01]) == pytest.approx(8 / 128)
    assert permutation_p([-0.01] * 7) == 1.0
    assert permutation_p([0.0] * 7) == 1.0


@pytest.mark.parametrize("ps,rejected", [
    ({"x": 0.01, "y": 0.02}, {"x", "y"}),
    ({"x": 0.01, "y": 0.04}, {"x", "y"}),
    ({"x": 0.03, "y": 0.04}, set()),
    ({"x": 0.02, "y": 0.2}, {"x"}),
    ({"x": 0.026, "y": 0.001}, {"x", "y"}),
    ({"x": 0.06, "y": 0.001}, {"y"}),
])
def test_holm_rejects_in_order_of_p_against_a_stepped_up_threshold_and_stops_at_the_first_miss(ps, rejected):
    assert holm(ps, 0.05) == rejected


def _net(arm, seed, optimal, certified=False):
    return {"run": "P", "arm": arm, "seed": seed, "positions": 1000, "optimal": optimal, "by_ply": {},
            "certified": certified, "failures": 0 if certified else 3}


def _readout(per_arm: dict):
    nets = [_net(arm, seed, opt) for arm, opts in per_arm.items() for seed, opt in zip(TEST_SPEC["seeds"], opts)]
    return {"measurement_fingerprint": TEST_SPEC["measurement_fp"], "plies": TEST_SPEC["plies"],
            "positions": TEST_SPEC["positions"],
            "runs": {f"P/{arm}": {"training_fingerprint": TEST_SPEC["era"],
                                  "config": {**cfg, "seeds": list(TEST_SPEC["seeds"]), "certify_depth": 6}}
                     for arm, cfg in TEST_SPEC["arms"].items()},
            "nets": nets}


BASE = [700, 690, 710, 705, 695, 700, 702]


def test_a_treatment_better_on_every_seed_is_supported_and_one_never_better_is_refuted():
    r = powered_report(_readout({"base": BASE, "x": [b + 30 for b in BASE], "y": [b - 5 for b in BASE]}), TEST_SPEC)
    assert r["integrity"] == []
    assert r["treatments"]["x"]["verdict"] == "supported" and r["treatments"]["x"]["p"] == pytest.approx(1 / 128)
    assert r["treatments"]["x"]["mean"] == pytest.approx(0.03)
    assert r["treatments"]["y"]["verdict"] == "refuted"


def test_a_positive_gain_the_test_does_not_reject_is_inconclusive():
    noisy = [b + d for b, d in zip(BASE, (30, -20, 25, -15, 10, -5, 2))]
    r = powered_report(_readout({"base": BASE, "x": noisy, "y": BASE}), TEST_SPEC)
    assert r["treatments"]["x"]["mean"] > 0 and r["treatments"]["x"]["verdict"] == "inconclusive"
    assert r["treatments"]["y"]["verdict"] == "refuted"


def test_holm_s_correction_is_applied_across_the_treatments():
    six_of_seven = [b + d for b, d in zip(BASE, (10, 10, 10, 10, 10, -3, 2))]
    r = powered_report(_readout({"base": BASE, "x": six_of_seven, "y": six_of_seven}), TEST_SPEC)
    assert r["treatments"]["x"]["p"] == pytest.approx(3 / 128)
    assert {t: v["verdict"] for t, v in r["treatments"].items()} == {"x": "supported", "y": "supported"}
    three = {**TEST_SPEC, "arms": {**TEST_SPEC["arms"], "z": {"a": 3}}, "treatments": ("x", "y", "z")}
    e = _readout({"base": BASE, "x": six_of_seven, "y": six_of_seven, "z": six_of_seven})
    e["runs"]["P/z"] = {"training_fingerprint": TEST_SPEC["era"],
                        "config": {"a": 3, "seeds": list(TEST_SPEC["seeds"]), "certify_depth": 6}}
    assert {v["verdict"] for v in powered_report(e, three)["treatments"].values()} == {"inconclusive"}


def test_certification_is_reported_per_arm_and_does_not_judge():
    e = _readout({"base": BASE, "x": [b + 30 for b in BASE], "y": BASE})
    e["nets"][0]["certified"] = True
    r = powered_report(e, TEST_SPEC)
    assert r["certified"] == {"base": 1, "x": 0, "y": 0} and r["treatments"]["x"]["verdict"] == "supported"


@pytest.mark.parametrize("breakage", ["fingerprint", "era", "config", "positions", "plies", "missing net",
                                      "extra seed"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _readout({"base": BASE, "x": BASE, "y": BASE})
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "era":
        e["runs"]["P/x"]["training_fingerprint"] = "0" * 12
    elif breakage == "config":
        e["runs"]["P/y"]["config"]["a"] = 9
    elif breakage == "positions":
        e["positions"] = 999
    elif breakage == "plies":
        e["plies"] = [0]
    elif breakage == "missing net":
        e["nets"] = e["nets"][1:]
    else:
        e["nets"].append(_net("base", 99, 700))
    r = powered_report(e, TEST_SPEC)
    assert r["integrity"] and {v["verdict"] for v in r["treatments"].values()} == {"not_run"}


def test_the_registered_comparison_is_base_against_the_three_earlier_fixes_on_seven_fresh_seeds():
    from harness.floor_c4_backplay import SPEC as T16_SPEC
    from harness.floor_c4_value_signal import SPEC as T14_SPEC

    assert SPEC["arms"] == {"base": T14_SPEC["arms"]["base"], "n_step": T14_SPEC["arms"]["n_step"],
                            "tree_value": T14_SPEC["arms"]["tree_value"], "backplay": T16_SPEC["arms"]["backplay"]}
    assert SPEC["treatments"] == ("n_step", "tree_value", "backplay")
    assert SPEC["seeds"] == tuple(range(431, 438)) and SPEC["prefix"] == "c49_T17"
    assert (SPEC["positions"], SPEC["plies"], SPEC["alpha"], SPEC["certify_depth"]) == (5142, [0, 2, 4, 6, 8], 0.05, 6)
