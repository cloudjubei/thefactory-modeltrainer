"""Direct tests for harness/floor_l5.py — L5, L2's rule reading the safe moves on unlevelled cells as a last level, on
L4's 32 positions, judged on completion as W6 and on its total against L4's control (L2's rule as it stands, no
library, already built)."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_l4 import SPEC as L4
from harness.floor_l4_control import SPEC as CONTROL
from harness.floor_l5 import SPEC, l5_report, l5_size
from harness.floor_w6 import w6_report


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_l5_is_l2_reading_unlevelled_cells_last_on_l4_s_positions():
    assert SPEC["indices"] == L4["indices"] and SPEC["roots"] == 32 and "library_from" not in SPEC
    assert SPEC["builder"] == {**L2["builder"], "unlevelled": "last"} and SPEC["search"] == L2["search"]
    assert SPEC["measurement_fp"] == "7717c79f94ea" != CONTROL["measurement_fp"] and SPEC["support"] == 0.8
    assert l5_report is w6_report


def test_l5_is_judged_against_l4_s_control_under_the_control_s_own_registration():
    control = _e(CONTROL, [100] + [0] * 31)
    assert l5_size(_e(SPEC, [80] + [0] * 31), control, SPEC)["verdict"] == "supported"
    assert l5_size(_e(SPEC, [100] + [0] * 31), control, SPEC)["verdict"] == "refuted"
    assert l5_size(_e(SPEC, [80] + [0] * 31), _e(CONTROL, [100] + [0] * 31, builder=SPEC["builder"]),
                   SPEC)["verdict"] == "not_run"
    assert l5_size(_e(SPEC, [80] + [0] * 31, builder=CONTROL["builder"]), control, SPEC)["verdict"] == "not_run"
