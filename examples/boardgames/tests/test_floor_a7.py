"""Direct tests for harness/floor_a7.py — A7, L5's rule with leaves kept already at 10x (not 30x), on F2's 16 heaviest
positions, judged on completion out of 16 and on its total against B1's control (L5 as it stands, the same positions,
no library, already built)."""
from __future__ import annotations

from harness.floor_a7 import SPEC, a7_report, a7_size
from harness.floor_b1 import SPEC as B1
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_k1 import SPEC as K1
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_a7_is_l5_accepting_leaves_at_10x_on_b1_s_positions_under_today_s_code():
    assert SPEC["indices"] == B1["indices"] and SPEC["roots"] == 16 and "library_from" not in SPEC
    assert (SPEC["complete_support"], SPEC["complete_refute"], SPEC["support"]) == (14, 12, 0.9)
    assert SPEC["builder"] == {**L5["builder"], "accept": 10} and L5["builder"]["accept"] == 30
    assert SPEC["search"] == L5["search"] and SPEC["measurement_fp"] == K1["measurement_fp"]
    assert a7_report is w6_report


def test_a7_is_judged_against_b1_s_control_under_the_control_s_own_registration():
    control = _e(CONTROL, [100] + [0] * 15)
    assert a7_size(_e(SPEC, [90] + [0] * 15), control)["verdict"] == "supported"
    assert a7_size(_e(SPEC, [91] + [0] * 15), control)["verdict"] == "inconclusive"
    assert a7_size(_e(SPEC, [100] + [0] * 15), control)["verdict"] == "refuted"
    assert a7_size(_e(SPEC, [90] + [0] * 15, incomplete=(207,)), control)["verdict"] == "inconclusive"
    assert a7_size(_e(SPEC, [90] + [0] * 15), _e(CONTROL, [100] + [0] * 15, builder=SPEC["builder"]))["verdict"] == (
        "not_run")
    assert a7_size(_e(SPEC, [90] + [0] * 15, builder=CONTROL["builder"]), control)["verdict"] == "not_run"
    assert a7_size(_e(SPEC, [90] + [0] * 15, measurement_fingerprint=L5["measurement_fp"]), control)["verdict"] == (
        "not_run")


def test_completion_is_judged_out_of_sixteen():
    def verdict(n):
        return a7_report(_e(SPEC, [1] * 16, incomplete=B1["indices"][n:]), SPEC)["complete"]["verdict"]
    assert (verdict(14), verdict(13), verdict(12)) == ("supported", "inconclusive", "refuted")
