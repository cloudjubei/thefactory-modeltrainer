"""Direct tests for harness/floor_l4.py and harness/floor_l4_control.py — L4, L2's rule preferring forcing moves, on 32
of F1's positions (its 16 heaviest and 16 spread over the rest), against a control built the same way without it;
judged on each arm's completion as W6 and on the forcing arm's total against the control's on the same positions."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_l4 import SPEC, l4_report, l4_size
from harness.floor_l4_control import SPEC as CONTROL
from harness.floor_w6 import w6_report


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_l4_is_l2_preferring_forcing_moves_on_f1_s_heaviest_and_spread_positions():
    assert SPEC["indices"][:16] == [218, 207, 113, 240, 487, 598, 61, 216, 467, 541, 1, 297, 652, 584, 358, 637]
    assert SPEC["indices"][16:] == [226, 530, 121, 15, 94, 324, 276, 548, 472, 449, 470, 568, 204, 159, 9, 353]
    assert SPEC["roots"] == 32 and SPEC["builder"] == {**L2["builder"], "forcing": True}
    assert SPEC["search"] == L2["search"] and "library_from" not in SPEC
    assert SPEC["measurement_fp"] == "1d12399d877c" != L2["measurement_fp"] and SPEC["support"] == 0.8
    assert l4_report is w6_report


def test_the_control_is_l4_without_forcing():
    assert CONTROL == {**SPEC, "builder": L2["builder"]}


def test_the_forcing_arm_is_judged_against_the_control():
    control = _e(CONTROL, [100] + [0] * 31)
    assert l4_size(_e(SPEC, [80] + [0] * 31), control, SPEC, CONTROL) == {
        "verdict": "supported", "bits": 80, "control_bits": 100, "ratio": 0.8, "lower_bound": False,
        "control_lower_bound": False}
    assert l4_size(_e(SPEC, [81] + [0] * 31), control, SPEC, CONTROL)["verdict"] == "inconclusive"
    assert l4_size(_e(SPEC, [99] + [0] * 31), control, SPEC, CONTROL)["verdict"] == "inconclusive"
    assert l4_size(_e(SPEC, [100] + [0] * 31), control, SPEC, CONTROL)["verdict"] == "refuted"
    r = l4_size(_e(SPEC, [10] * 32), _e(CONTROL, [20] * 32), SPEC, CONTROL)
    assert (r["bits"], r["control_bits"]) == (320, 640)


def test_an_unfinished_forcing_root_can_only_refute_and_an_unfinished_control_root_only_support():
    control = _e(CONTROL, [100] + [0] * 31)
    assert l4_size(_e(SPEC, [1] * 32, incomplete={218}), control, SPEC, CONTROL)["verdict"] == "inconclusive"
    assert l4_size(_e(SPEC, [100] + [0] * 31, incomplete={218}), control, SPEC, CONTROL)["verdict"] == "refuted"
    partial = _e(CONTROL, [100] + [0] * 31, incomplete={353})
    assert l4_size(_e(SPEC, [100] + [0] * 31), partial, SPEC, CONTROL)["verdict"] == "inconclusive"
    r = l4_size(_e(SPEC, [80] + [0] * 31), partial, SPEC, CONTROL)
    assert r["verdict"] == "supported" and r["control_lower_bound"] is True


def test_a_run_off_its_registration_is_not_judged():
    control = _e(CONTROL, [10] * 32)
    for bad in (_e(SPEC, [10] * 32, measurement_fingerprint=L2["measurement_fp"]), _e(SPEC, [10] * 31),
                _e(SPEC, [10] * 32, builder=L2["builder"])):
        assert l4_size(bad, control, SPEC, CONTROL)["verdict"] == "not_run"
    for bad in (_e(CONTROL, [10] * 32, builder=SPEC["builder"]), _e(CONTROL, [10] * 31)):
        assert l4_size(_e(SPEC, [10] * 32), bad, SPEC, CONTROL)["verdict"] == "not_run"
