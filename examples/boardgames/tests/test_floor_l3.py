"""Direct tests for harness/floor_l3.py — L3, L2 with two-ply safety (a move is not safe when the opponent can answer
it with a forced win in two) on W6's 32 positions, judged on completion as W6 and on the 32 strategies' total against
L2's 14,666 bits on the same positions."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_l3 import SPEC, l3_report, l3_size
from harness.floor_w6 import w6_report


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(SPEC["indices"], sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_l3_is_l2_with_two_ply_safety_on_the_same_positions():
    assert SPEC["indices"] == L2["indices"] and SPEC["roots"] == 32
    assert SPEC["builder"] == {**L2["builder"], "safe_depth": 2} and SPEC["search"] == L2["search"]
    assert SPEC["measurement_fp"] == "081011f9bbfa" != L2["measurement_fp"]
    assert (SPEC["l2_bits"], SPEC["support"]) == (14_666, 0.8)
    assert l3_report is w6_report


def test_the_size_is_judged_against_l2_s():
    spec = {**SPEC, "l2_bits": 100}
    r = l3_size(_e([80] + [0] * 31), spec)
    assert r == {"verdict": "supported", "bits": 80, "ratio": 0.8, "lower_bound": False}
    assert l3_size(_e([81] + [0] * 31), spec)["verdict"] == "inconclusive"
    assert l3_size(_e([99] + [0] * 31), spec)["verdict"] == "inconclusive"
    assert l3_size(_e([100] + [0] * 31), spec)["verdict"] == "refuted"
    assert l3_size(_e([10] * 32), spec)["bits"] == 320


def test_an_unfinished_root_can_only_refute():
    spec = {**SPEC, "l2_bits": 100}
    assert l3_size(_e([1] * 32, incomplete={471}), spec)["verdict"] == "inconclusive"
    assert l3_size(_e([100] + [0] * 31, incomplete={471}), spec)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([10] * 32, measurement_fingerprint=L2["measurement_fp"]), _e([10] * 31),
                _e([10] * 32, builder=L2["builder"])):
        assert l3_size(bad, SPEC)["verdict"] == "not_run"
