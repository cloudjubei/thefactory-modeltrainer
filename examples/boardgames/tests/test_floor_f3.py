"""Direct tests for harness/floor_f3.py — F3, F2 with L6's rule (a deciding level's ties broken by tactics): every
frontier position in F1's order and waves from the same seeded library, judged on completion as F1 and on the whole
strategy's size against F2's 455,959 bits."""
from __future__ import annotations

from harness.floor_f1 import f1_complete
from harness.floor_f2 import SPEC as F2
from harness.floor_f3 import SPEC, f3_complete, f3_size
from harness.floor_k1 import SPEC as K1
from harness.floor_l6 import SPEC as L6

BASE = 204 * 4 + 259 + 8_618


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(SPEC["indices"], sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_f3_is_f2_with_l6_s_rule_under_today_s_code():
    assert {k: v for k, v in SPEC.items() if k not in ("builder", "measurement_fp", "f2_bits")} == {
        k: v for k, v in F2.items() if k not in ("builder", "measurement_fp", "f1_bits")}
    assert SPEC["builder"] == {**F2["builder"], "order": "tactics"} == L6["builder"]
    assert SPEC["measurement_fp"] == K1["measurement_fp"] != F2["measurement_fp"]
    assert (SPEC["f2_bits"], SPEC["support"], SPEC["wave_size"]) == (455_959, (9, 10), 32)
    assert SPEC["library_from"] == F2["library_from"] and SPEC["library_bits"] == F2["library_bits"] == 8_618
    assert f3_complete is f1_complete


def test_the_whole_strategy_is_judged_against_f2_s():
    small = (9 * 455_959 - 10 * BASE) // 10
    assert f3_size(_e([small] + [0] * 670), SPEC) == {"verdict": "supported", "bits": BASE + small,
                                                     "lower_bound": False}
    assert f3_size(_e([small + 1] + [0] * 670), SPEC)["verdict"] == "inconclusive"
    assert f3_size(_e([455_959 - BASE - 1] + [0] * 670), SPEC)["verdict"] == "inconclusive"
    assert f3_size(_e([455_959 - BASE] + [0] * 670), SPEC)["verdict"] == "refuted"


def test_an_unfinished_position_can_only_refute():
    assert f3_size(_e([1] * 671, incomplete={362}), SPEC) == {"verdict": "inconclusive", "bits": BASE + 671,
                                                             "lower_bound": True}
    assert f3_size(_e([455_959 - BASE] + [0] * 670, incomplete={362}), SPEC)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([1] * 671, measurement_fingerprint=F2["measurement_fp"]), _e([1] * 670),
                _e([1] * 671, builder=F2["builder"])):
        assert f3_size(bad, SPEC)["verdict"] == "not_run"
        assert f3_complete(bad, SPEC)["verdict"] == "not_run"
