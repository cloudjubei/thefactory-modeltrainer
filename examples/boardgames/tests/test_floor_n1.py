"""Direct tests for harness/floor_n1.py — N1, a learned default paid once: a small net trained on F2's positions away
from L4's 32 roots, scored on D1's test positions, judged on its misses against the tactics order's."""
from __future__ import annotations

from harness.floor_l4 import SPEC as L4
from harness.floor_n1 import SPEC, n1_misses


def _e(hits=4_423, bits=19_996, **over):
    totals = {"solved": 4_523, "lowest": 4_215, "tactics": 4_323, "net": hits, "net_float": hits, "bits": bits}
    return {"measurement_fingerprint": SPEC["measurement_fp"], "test": SPEC["test"], "train": SPEC["train"],
            "net": SPEC["net"], "totals": {**totals, **over.pop("totals", {})}, **over}


def test_n1_tests_on_d1_s_positions_and_trains_on_f2_s_others():
    assert SPEC["test"] == {"spec": "harness.floor_l5", "targets": "c49_l5.json.gz", "per_leaf": 400, "per_root": 150,
                            "seed": 0}
    assert SPEC["train"] == {"spec": "harness.floor_f2", "targets": "c49_f2.json.gz", "per_leaf": 400,
                             "per_root": 150, "seed": 1, "skip": L4["indices"]}
    assert SPEC["net"] == {"hidden": 32, "weight_bits": 4, "epochs": 40, "batch": 256, "lr": 0.001, "seed": 0}
    assert SPEC["d1"] == {"solved": 4_523, "lowest": 4_215, "tactics": 4_323}
    assert (SPEC["max_bits"], SPEC["support"], SPEC["refute"]) == (20_000, 100, 200)


def test_the_net_is_judged_on_its_misses_against_the_tactics_order():
    assert n1_misses(_e(4_423)) == {"verdict": "supported", "misses": 100, "tactics_misses": 200, "bits": 19_996}
    assert n1_misses(_e(4_422))["verdict"] == "inconclusive"
    assert n1_misses(_e(4_324))["verdict"] == "inconclusive"
    assert n1_misses(_e(4_323))["verdict"] == "refuted"


def test_the_quantized_net_is_judged_not_the_float_one():
    assert n1_misses(_e(4_323, totals={"net_float": 4_523}))["verdict"] == "refuted"


def test_a_measurement_off_its_registration_is_not_judged():
    for bad in (_e(measurement_fingerprint="x"), _e(test={**SPEC["test"], "seed": 1}),
                _e(train={**SPEC["train"], "skip": []}), _e(net={**SPEC["net"], "hidden": 64}),
                _e(totals={"solved": 4_522}), _e(totals={"tactics": 4_322}), _e(totals={"lowest": 4_216}),
                _e(bits=20_001)):
        assert n1_misses(bad) == {"verdict": "not_run"}
    assert n1_misses(_e(bits=20_000))["verdict"] == "supported"
