"""Direct tests for harness/floor_a6.py — A6, L5's rule with leaves kept only at 100x (not 30x), on F2's 16 heaviest
positions, judged on completion out of 16 and on its total against B1's control (L5 as it stands, the same positions,
no library, already built)."""
from __future__ import annotations

from harness.floor_a6 import SPEC, a6_report, a6_size
from harness.floor_b1 import SPEC as B1
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_a6_is_l5_accepting_leaves_at_100x_on_b1_s_positions():
    assert SPEC["indices"] == B1["indices"] and SPEC["roots"] == 16
    assert (SPEC["complete_support"], SPEC["complete_refute"], SPEC["support"]) == (14, 12, 0.9)
    assert SPEC["builder"] == {**L5["builder"], "accept": 100} and L5["builder"]["accept"] == 30
    assert SPEC["search"] == L5["search"] and SPEC["measurement_fp"] == L5["measurement_fp"]
    assert a6_report is w6_report


def test_a6_is_judged_against_b1_s_control_under_the_control_s_own_registration():
    control = _e(CONTROL, [100] + [0] * 15)
    assert a6_size(_e(SPEC, [90] + [0] * 15), control)["verdict"] == "supported"
    assert a6_size(_e(SPEC, [91] + [0] * 15), control)["verdict"] == "inconclusive"
    assert a6_size(_e(SPEC, [100] + [0] * 15), control)["verdict"] == "refuted"
    assert a6_size(_e(SPEC, [90] + [0] * 15), _e(CONTROL, [100] + [0] * 15, builder=SPEC["builder"]))["verdict"] == (
        "not_run")
    assert a6_size(_e(SPEC, [90] + [0] * 15, builder=CONTROL["builder"]), control)["verdict"] == "not_run"
