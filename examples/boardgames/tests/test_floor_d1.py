"""Direct tests for harness/floor_d1.py — D1, which zero-bit default to give the leaf rule next: the "tactics" order
judged on its misses against the lowest's, on positions sampled from L5's certified leaves."""
from __future__ import annotations

from fractions import Fraction

from harness.floor_d1 import SPEC, d1_misses


def _e(solved, lowest_hits, tactics_hits, **over):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "spec": SPEC["spec"], "targets": SPEC["targets"],
            "per_root": SPEC["per_root"], "per_leaf": SPEC["per_leaf"], "seed": SPEC["seed"],
            "totals": {"solved": solved, "hits": {"lowest": lowest_hits, "tactics": tactics_hits, "highest": 0}},
            **over}


def test_d1_measures_l5_s_leaves():
    assert SPEC["measurement_fp"] == "576f1422643b"
    assert (SPEC["spec"], SPEC["targets"]) == ("harness.floor_l5", "c49_l5.json.gz")
    assert (SPEC["per_root"], SPEC["per_leaf"], SPEC["seed"], SPEC["order"]) == (150, 400, 0, "tactics")
    assert (SPEC["support"], SPEC["min_misses"]) == (Fraction(2, 3), 30)


def test_the_order_s_misses_are_judged_against_the_lowest_s():
    assert d1_misses(_e(1000, 940, 960), SPEC) == {"verdict": "supported", "lowest_misses": 60, "misses": 40,
                                                   "ratio": 0.666667}
    assert d1_misses(_e(1000, 940, 959), SPEC)["verdict"] == "refuted"
    assert d1_misses(_e(1000, 940, 940), SPEC)["verdict"] == "refuted"
    assert d1_misses(_e(1000, 940, 1000), SPEC)["verdict"] == "supported"


def test_the_judged_order_is_the_spec_s():
    assert d1_misses(_e(1000, 940, 1000), {**SPEC, "order": "highest"})["verdict"] == "refuted"


def test_too_few_misses_of_the_lowest_cannot_judge():
    assert d1_misses(_e(1000, 970, 1000), SPEC) == {"verdict": "supported", "lowest_misses": 30, "misses": 0,
                                                     "ratio": 0.0}
    assert d1_misses(_e(1000, 971, 1000), SPEC) == {"verdict": "inconclusive", "lowest_misses": 29, "misses": 0}


def test_another_measurement_is_not_judged():
    for over in ({"measurement_fingerprint": "x"}, {"spec": "harness.floor_l2"}, {"targets": "c49_f2.json.gz"},
                 {"per_root": 20}, {"per_leaf": 200}, {"seed": 1}):
        assert d1_misses(_e(1000, 900, 1000, **over), SPEC)["verdict"] == "not_run"
