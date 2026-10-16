"""Direct tests for harness/floor_m1.py — M1, F1's maps under exact cheaper codes: judged on the share of F1's map bits
the cheapest code per map (plus its selector) saves, the measurement first reproducing F1's own map bits."""
from __future__ import annotations

from fractions import Fraction

from harness.floor_m1 import SPEC, m1_saving


def _e(current, best, **over):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "source": SPEC["source"], "maps": SPEC["maps"],
            "totals": {"bitmap": current, "best": best}, **over}


def test_m1_measures_f1_s_maps():
    assert SPEC["measurement_fp"] == "133c5f0ac4b4" and SPEC["source"] == "c49_f1.json.gz"
    assert (SPEC["maps"], SPEC["f1_map_bits"], SPEC["n_levels"]) == (3_196, 292_054, 8)
    assert (SPEC["support"], SPEC["refute"]) == (Fraction(1, 10), Fraction(1, 20))


def test_the_saving_is_judged_against_f1_s_map_bits():
    spec = {**SPEC, "f1_map_bits": 1000}
    assert m1_saving(_e(1000, 900), spec) == {"verdict": "supported", "saving": 0.1, "bits": 900}
    assert m1_saving(_e(1000, 901), spec)["verdict"] == "inconclusive"
    assert m1_saving(_e(1000, 950), spec)["verdict"] == "inconclusive"
    assert m1_saving(_e(1000, 951), spec)["verdict"] == "refuted"


def test_a_measurement_that_does_not_reproduce_f1_s_map_bits_is_not_judged():
    assert m1_saving(_e(292_055, 1), SPEC)["verdict"] == "not_run"
    assert m1_saving(_e(292_054, 1, measurement_fingerprint="x"), SPEC)["verdict"] == "not_run"
    assert m1_saving(_e(292_054, 1, source="c49_w9.json.gz"), SPEC)["verdict"] == "not_run"
    assert m1_saving(_e(292_054, 1, maps=3_195), SPEC)["verdict"] == "not_run"
