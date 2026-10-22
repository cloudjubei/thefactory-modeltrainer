"""Direct tests for harness/floor_r1.py — R1, F2's leaf references under first_use: judged on the share of F2's whole
455,959 bits it saves, the measurement first reproducing F2's own reference bits."""
from __future__ import annotations

from fractions import Fraction

from harness.floor_r1 import SPEC, r1_saving


def _e(fixed, first_use, **over):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "source": SPEC["source"], "roots": SPEC["roots"],
            "totals": {"fixed": fixed, "first_use": first_use}, **over}


def test_r1_measures_f2_s_references():
    assert SPEC["measurement_fp"] == "f62cfb41901f" and SPEC["source"] == "c49_f2.json.gz"
    assert (SPEC["roots"], SPEC["f2_bits"], SPEC["f2_reference_bits"]) == (671, 455_959, 100_888)
    assert (SPEC["support"], SPEC["refute"]) == (Fraction(1, 10), Fraction(1, 20))


def test_the_saving_is_judged_against_f2_s_whole_strategy():
    spec = {**SPEC, "f2_bits": 1000, "f2_reference_bits": 300}
    assert r1_saving(_e(300, 200), spec) == {"verdict": "supported", "saving": 0.1, "bits": 900, "references": 200}
    assert r1_saving(_e(300, 201), spec)["verdict"] == "inconclusive"
    assert r1_saving(_e(300, 250), spec)["verdict"] == "inconclusive"
    assert r1_saving(_e(300, 251), spec)["verdict"] == "refuted"
    assert r1_saving(_e(300, 400), spec)["verdict"] == "refuted"


def test_a_measurement_that_does_not_reproduce_f2_s_references_is_not_judged():
    assert r1_saving(_e(100_889, 1), SPEC)["verdict"] == "not_run"
    assert r1_saving(_e(100_888, 1, measurement_fingerprint="x"), SPEC)["verdict"] == "not_run"
    assert r1_saving(_e(100_888, 1, source="c49_f1.json.gz"), SPEC)["verdict"] == "not_run"
    assert r1_saving(_e(100_888, 1, roots=670), SPEC)["verdict"] == "not_run"
    assert r1_saving(_e(100_888, 1), SPEC)["verdict"] == "supported"
