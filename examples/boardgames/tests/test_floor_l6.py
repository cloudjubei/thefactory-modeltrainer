"""Direct tests for harness/floor_l6.py — L6, L5's rule breaking a deciding level's ties by tactics, on L4's 32
positions, judged on completion as W6 and on its total against L5 (already built, under its own registration)."""
from __future__ import annotations

from harness.floor_l4 import SPEC as L4
from harness.floor_l5 import SPEC as L5
from harness.floor_l6 import SPEC, l6_report, l6_size
from harness.floor_w6 import w6_report


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_l6_is_l5_breaking_ties_by_tactics_on_l4_s_positions():
    assert SPEC["indices"] == L4["indices"] and SPEC["roots"] == 32 and "library_from" not in SPEC
    assert SPEC["builder"] == {**L5["builder"], "order": "tactics"} and SPEC["search"] == L5["search"]
    assert SPEC["measurement_fp"] == "2285bc9e9ca8" != L5["measurement_fp"] and SPEC["support"] == 0.9
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (28, 24) and l6_report is w6_report


def test_l6_is_judged_against_l5_under_l5_s_own_registration():
    control = _e(L5, [100] + [0] * 31)
    assert l6_size(_e(SPEC, [90] + [0] * 31), control, SPEC)["verdict"] == "supported"
    assert l6_size(_e(SPEC, [91] + [0] * 31), control, SPEC)["verdict"] == "inconclusive"
    assert l6_size(_e(SPEC, [100] + [0] * 31), control, SPEC)["verdict"] == "refuted"
    assert l6_size(_e(SPEC, [90] + [0] * 31), _e(L5, [100] + [0] * 31, builder=SPEC["builder"]),
                   SPEC)["verdict"] == "not_run"
    assert l6_size(_e(SPEC, [90] + [0] * 31, builder=L5["builder"]), control, SPEC)["verdict"] == "not_run"
