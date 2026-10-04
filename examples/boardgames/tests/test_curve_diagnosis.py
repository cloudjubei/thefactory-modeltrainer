"""Direct tests for harness/curve_diagnosis.py — T20: is the Connect-4 process still learning at iteration 20? Each
seed's net is scored on the fixed position set after every training pass; the share late in a long run is compared
with the share at the pilots' length, for the opening plies and for all positions."""
from __future__ import annotations

import pytest

from harness.curve_diagnosis import SPEC, curve_report

TEST_SPEC = {"config": {"a": 1, "iterations": 10}, "seeds": (1, 2), "era": "e" * 12, "measurement_fp": "m" * 12,
             "early": [3, 4], "late": [9, 10], "opening": ["0", "2"], "support_at": 0.03, "refute_at": 0.01}


def _point(opening: float, overall: float, n: int = 100) -> dict:
    o = round(opening * n)
    rest = round(overall * 2 * n) - o
    return {"positions": 2 * n, "optimal": o + rest,
            "by_ply": {"0": {"positions": n // 2, "optimal": o // 2},
                       "2": {"positions": n - n // 2, "optimal": o - o // 2},
                       "6": {"positions": n, "optimal": rest}}}


def _seed(seed, opening_curve, overall_curve):
    return {"seed": seed, "curve": [_point(o, a) for o, a in zip(opening_curve, overall_curve)]}


def _run(seed, opening_gain, overall_gain, curve_len):
    opening = [0.6] * 5 + [0.6 + opening_gain] * (curve_len - 5)
    overall = [0.7] * 5 + [0.7 + overall_gain] * (curve_len - 5)
    return {"training_fingerprint": TEST_SPEC["era"], "measurement_fingerprint": TEST_SPEC["measurement_fp"],
            "config": {**TEST_SPEC["config"], "seeds": [seed]}, "seeds": [_seed(seed, opening, overall)]}


def _evidence(opening_gain=0.05, overall_gain=0.05, curve_len=11):
    return [_run(s, opening_gain, overall_gain, curve_len) for s in TEST_SPEC["seeds"]]


@pytest.mark.parametrize("gain,verdict", [(0.05, "supported"), (0.03, "supported"), (0.02, "inconclusive"),
                                          (0.01, "refuted"), (-0.02, "refuted")])
def test_the_opening_reading_is_the_pooled_late_minus_early_share(gain, verdict):
    r = curve_report(_evidence(opening_gain=gain, overall_gain=0.0), TEST_SPEC)
    assert r["integrity"] == [] and r["opening"]["verdict"] == verdict
    assert r["opening"]["gain"] == pytest.approx(gain) and r["overall"]["verdict"] == "refuted"


def test_the_overall_reading_is_judged_on_its_own():
    r = curve_report(_evidence(opening_gain=0.0, overall_gain=0.04), TEST_SPEC)
    assert r["overall"]["verdict"] == "supported" and r["overall"]["gain"] == pytest.approx(0.04)
    assert r["opening"]["verdict"] == "refuted"


def test_early_and_late_are_the_registered_passes_averaged_per_seed():
    e = _evidence(opening_gain=0.0, overall_gain=0.0)
    for row in (r for run in e for r in run["seeds"]):
        row["curve"][2] = _point(0.7, 0.7)
        row["curve"][8] = _point(0.8, 0.7)
    r = curve_report(e, TEST_SPEC)
    assert r["opening"]["gain"] == pytest.approx(0.05)
    assert r["opening"]["seeds"] == {"1": {"early": pytest.approx(0.65), "late": pytest.approx(0.7)},
                                     "2": {"early": pytest.approx(0.65), "late": pytest.approx(0.7)}}


@pytest.mark.parametrize("breakage", ["era", "fingerprint", "config", "seeds", "seed mislabelled", "short curve"])
def test_evidence_that_is_not_the_registered_run_is_not_run(breakage):
    e = _evidence(curve_len=10 if breakage == "short curve" else 11)
    if breakage == "era":
        e[1]["training_fingerprint"] = "0" * 12
    elif breakage == "fingerprint":
        e[0]["measurement_fingerprint"] = "0" * 12
    elif breakage == "config":
        e[1]["config"]["a"] = 2
    elif breakage == "seeds":
        e = e[:1]
    elif breakage == "seed mislabelled":
        e[0]["config"]["seeds"] = [5]
    r = curve_report(e, TEST_SPEC)
    assert r["integrity"] and r["opening"] == {"verdict": "not_run"} and r["overall"] == {"verdict": "not_run"}


def test_the_registered_diagnostic_runs_the_base_pilot_three_times_as_long_on_three_fresh_seeds():
    from harness.floor_c4_value_signal import SPEC as T14_SPEC

    assert SPEC["config"] == {**T14_SPEC["arms"]["base"], "iterations": 60}
    assert SPEC["seeds"] == (471, 472, 473)
    assert (SPEC["early"], SPEC["late"]) == ([16, 17, 18, 19, 20], [56, 57, 58, 59, 60])
    assert SPEC["opening"] == ["0", "2", "4"] and (SPEC["support_at"], SPEC["refute_at"]) == (0.03, 0.01)
