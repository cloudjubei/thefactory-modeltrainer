"""Direct tests for harness/floor_q1.py — Q1, a seeded sample of F2's leaves with exceptions: judged on the share of
their coded exception bits that flags coded per class, and moves coded tactics first, save."""
from __future__ import annotations

from fractions import Fraction

from harness.floor_q1 import SPEC, q1_flags, q1_moves


def _e(coded=1000, one_class=900, per_class=900, plain=100, tactics_first=100, **over):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "source": SPEC["source"], "seed": SPEC["seed"],
            "leaves": SPEC["leaves"], "totals": {"coded": coded, "one_class": one_class, "per_class": per_class,
                                                 "plain": plain, "tactics_first": tactics_first}, **over}


def test_q1_samples_sixty_of_f2_s_leaves_with_exceptions():
    assert SPEC["source"] == "c49_f2.json.gz" and (SPEC["seed"], SPEC["leaves"]) == (0, 60)
    assert SPEC["measurement_fp"] == "0183157a95c4" and SPEC["support"] == Fraction(1, 10)


def test_flags_by_class_are_judged_on_the_share_of_the_exception_bits_they_save():
    assert q1_flags(_e(per_class=800)) == {"verdict": "supported", "saving": 0.1}
    assert q1_flags(_e(per_class=801))["verdict"] == "inconclusive"
    assert q1_flags(_e(per_class=899))["verdict"] == "inconclusive"
    assert q1_flags(_e(per_class=900))["verdict"] == "refuted"
    assert q1_flags(_e(per_class=950)) == {"verdict": "refuted", "saving": -0.05}


def test_moves_tactics_first_are_judged_the_same_way():
    assert q1_moves(_e(tactics_first=0)) == {"verdict": "supported", "saving": 0.1}
    assert q1_moves(_e(tactics_first=1))["verdict"] == "inconclusive"
    assert q1_moves(_e(tactics_first=99))["verdict"] == "inconclusive"
    assert q1_moves(_e(tactics_first=100))["verdict"] == "refuted"


def test_a_measurement_off_its_registration_is_not_judged():
    for bad in (_e(measurement_fingerprint="x"), _e(source="c49_l5.json.gz"), _e(seed=1), _e(leaves=59),
                _e(coded=999)):
        assert q1_flags(bad) == {"verdict": "not_run"} and q1_moves(bad) == {"verdict": "not_run"}
