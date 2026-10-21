"""Direct tests for harness/floor_k1.py — K1, L6's rule with two-phase maps allowed, on L4's 32 positions, judged on
completion as W6, on its total against L6 and on F1's 16 heaviest of the 32 against L6's same 16."""
from __future__ import annotations

from harness.floor_l4 import SPEC as L4
from harness.floor_l6 import SPEC as L6
from harness.floor_k1 import SPEC, k1_heavy, k1_report, k1_size
from harness.floor_w6 import w6_report

HEAVY = L4["indices"][:16]


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def _sizes(heavy, light):
    return [heavy] * 16 + [light] * 16


def test_k1_is_l6_with_two_phase_maps_on_l4_s_positions():
    assert SPEC["indices"] == L4["indices"] and SPEC["roots"] == 32 and "library_from" not in SPEC
    assert SPEC["builder"] == {**L6["builder"], "phases": True} and SPEC["search"] == L6["search"]
    assert SPEC["measurement_fp"] != L6["measurement_fp"] and SPEC["support"] == 0.9
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (28, 24) and k1_report is w6_report
    assert SPEC["heavy"] == HEAVY == [218, 207, 113, 240, 487, 598, 61, 216, 467, 541, 1, 297, 652, 584, 358, 637]


def test_k1_s_total_is_judged_against_l6_under_l6_s_own_registration():
    control = _e(L6, _sizes(10, 0))
    assert k1_size(_e(SPEC, _sizes(9, 0)), control, SPEC)["verdict"] == "supported"
    assert k1_size(_e(SPEC, _sizes(10, 0)), control, SPEC)["verdict"] == "refuted"
    assert k1_size(_e(SPEC, _sizes(9, 0), builder=L6["builder"]), control, SPEC)["verdict"] == "not_run"


def test_the_heavy_sixteen_are_judged_against_l6_s_same_sixteen_alone():
    control = _e(L6, _sizes(10, 50))
    assert k1_heavy(_e(SPEC, _sizes(9, 500)), control, SPEC) == {
        "verdict": "supported", "bits": 144, "control_bits": 160, "ratio": 0.9, "lower_bound": False,
        "control_lower_bound": False}
    assert k1_heavy(_e(SPEC, [9] * 15 + [10] + [0] * 16), control, SPEC)["verdict"] == "inconclusive"
    assert k1_heavy(_e(SPEC, _sizes(10, 0)), control, SPEC)["verdict"] == "refuted"


def test_an_unfinished_heavy_root_only_bounds_the_arm_and_an_unfinished_light_one_does_not_count():
    control = _e(L6, _sizes(10, 50))
    assert k1_heavy(_e(SPEC, _sizes(8, 0), incomplete=(218,)), control, SPEC)["verdict"] == "inconclusive"
    assert k1_heavy(_e(SPEC, _sizes(8, 0), incomplete=(226,)), control, SPEC)["verdict"] == "supported"
    assert k1_heavy(_e(SPEC, _sizes(8, 0)), _e(L6, _sizes(10, 50), incomplete=(218,)),
                    SPEC)["verdict"] == "supported"
    assert k1_heavy(_e(SPEC, _sizes(11, 0)), _e(L6, _sizes(10, 50), incomplete=(218,)),
                    SPEC)["verdict"] == "inconclusive"


def test_the_heavy_judge_needs_both_runs_as_registered():
    control = _e(L6, _sizes(10, 50))
    assert k1_heavy(_e(SPEC, _sizes(8, 0), builder=L6["builder"]), control, SPEC)["verdict"] == "not_run"
    assert k1_heavy(_e(SPEC, _sizes(8, 0)), _e(L6, _sizes(10, 50), builder=SPEC["builder"]),
                    SPEC)["verdict"] == "not_run"
