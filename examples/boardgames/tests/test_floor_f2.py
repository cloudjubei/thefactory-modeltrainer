"""Direct tests for harness/floor_f2.py — F2, F1 with L5's rule (the safe moves on unlevelled cells a last level): every
frontier position in F1's order and waves from the same seeded library, judged on completion as F1 and on the whole
strategy's size against F1's 548,424 bits."""
from __future__ import annotations

from harness.floor_f1 import SPEC as F1, f1_complete
from harness.floor_f2 import SPEC, f2_complete, f2_size
from harness.floor_l5 import SPEC as L5

BASE = 204 * 4 + 259 + 8_618


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(SPEC["indices"], sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_f2_is_f1_with_l5_s_rule():
    assert {k: v for k, v in SPEC.items() if k not in ("builder", "measurement_fp", "f1_bits", "support")} == {
        k: v for k, v in F1.items() if k not in ("builder", "measurement_fp")}
    assert SPEC["builder"] == {**F1["builder"], "unlevelled": "last"} == L5["builder"]
    assert SPEC["measurement_fp"] == L5["measurement_fp"] == "7717c79f94ea"
    assert (SPEC["f1_bits"], SPEC["support"]) == (548_424, (9, 10))
    assert f2_complete is f1_complete


def test_the_whole_strategy_is_judged_against_f1_s():
    small = (4_935_816 - 10 * BASE) // 10
    assert f2_size(_e([small] + [0] * 670), SPEC) == {"verdict": "supported", "bits": BASE + small, "lower_bound": False}
    assert f2_size(_e([small + 1] + [0] * 670), SPEC)["verdict"] == "inconclusive"
    assert f2_size(_e([548_424 - BASE - 1] + [0] * 670), SPEC)["verdict"] == "inconclusive"
    assert f2_size(_e([548_424 - BASE] + [0] * 670), SPEC)["verdict"] == "refuted"


def test_exactly_nine_tenths_supports():
    spec = {**SPEC, "f1_bits": 20_000}
    assert f2_size(_e([18_000 - BASE] + [0] * 670), spec)["verdict"] == "supported"
    assert f2_size(_e([18_001 - BASE] + [0] * 670), spec)["verdict"] == "inconclusive"


def test_an_unfinished_position_can_only_refute():
    assert f2_size(_e([1] * 671, incomplete={362}), SPEC) == {"verdict": "inconclusive", "bits": BASE + 671,
                                                             "lower_bound": True}
    assert f2_size(_e([548_424 - BASE] + [0] * 670, incomplete={362}), SPEC)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([1] * 671, measurement_fingerprint=F1["measurement_fp"]), _e([1] * 670),
                _e([1] * 671, builder=F1["builder"])):
        assert f2_size(bad, SPEC)["verdict"] == "not_run"
