"""Direct tests for harness/floor_b1.py and harness/floor_b1_control.py — B1, L5's rule with deeper leaf searches of
120 s instead of 30 s, on F2's 16 heaviest positions, against a control built the same way with 30 s; judged on each
arm's completion (14 of 16 supports, 12 or fewer refutes) and on the arm's total against the control's."""
from __future__ import annotations

from harness.floor_b1 import SPEC, b1_report, b1_size
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_b1_is_l5_with_longer_deeper_searches_on_f2_s_heaviest_positions():
    assert SPEC["indices"] == [207, 113, 216, 240, 218, 467, 16, 622, 605, 1, 487, 584, 58, 541, 542, 212]
    assert SPEC["roots"] == 16 and (SPEC["complete_support"], SPEC["complete_refute"]) == (14, 12)
    assert SPEC["builder"] == {**L5["builder"], "budgets": [[0, 1200.0], [2, 120.0]]}
    assert SPEC["search"] == L5["search"] and SPEC["measurement_fp"] == L5["measurement_fp"]
    assert "library_from" not in SPEC and SPEC["support"] == 0.9
    assert b1_report is w6_report


def test_the_control_is_b1_with_l5_s_budgets():
    assert CONTROL == {**SPEC, "builder": L5["builder"]}


def test_completion_is_judged_out_of_sixteen():
    assert b1_report(_e(SPEC, [1] * 16, incomplete={207, 113}), SPEC)["complete"] == {"verdict": "supported", "count": 14}
    assert b1_report(_e(SPEC, [1] * 16, incomplete={207, 113, 216}), SPEC)["complete"]["verdict"] == "inconclusive"
    assert b1_report(_e(SPEC, [1] * 16, incomplete={207, 113, 216, 240}), SPEC)["complete"]["verdict"] == "refuted"


def test_the_arm_is_judged_against_the_control_at_nine_tenths():
    control = _e(CONTROL, [100] + [0] * 15)
    assert b1_size(_e(SPEC, [90] + [0] * 15), control)["verdict"] == "supported"
    assert b1_size(_e(SPEC, [91] + [0] * 15), control)["verdict"] == "inconclusive"
    assert b1_size(_e(SPEC, [100] + [0] * 15), control)["verdict"] == "refuted"
    assert b1_size(_e(SPEC, [90] + [0] * 15), _e(CONTROL, [100] + [0] * 15, builder=SPEC["builder"]))["verdict"] == (
        "not_run")
