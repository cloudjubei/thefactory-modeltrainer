"""Direct tests for harness/floor_w2.py — W2, W1's projection redone with root searches, honest budgets, walk-order
exceptions and a map library shared across two waves. Synthetic evidence for the library measure."""
from __future__ import annotations

from harness.floor_p2 import SPEC as P2
from harness.floor_w1 import SPEC as W1, w1_report
from harness.floor_w2 import SPEC, library_report, w2_report


def test_w2_rebuilds_w1_s_sample_in_order_with_p2_s_builder_and_w1_s_cap():
    assert SPEC["indices"] == [362, 87, 316, 346, 230, 323, 393, 263] and SPEC["wave_size"] == 4
    assert SPEC["builder"] == {**P2["builder"], "seconds": 7200.0}
    assert {k: SPEC[k] for k in ("ply", "roots", "source", "frontier", "positions_fp", "search")} == \
        {k: W1[k] for k in ("ply", "roots", "source", "frontier", "positions_fp", "search")}
    assert (SPEC["projection_support"], SPEC["projection_refute"]) == (3_982_628, 7_965_257)


def test_w2_is_projected_as_w1():
    assert w2_report is w1_report


def _root(shared, leaves, used):
    return {"shared": shared, "bits": {"leaves": leaves}, "library_leaves": used}


def test_the_library_share_counts_only_builds_that_had_a_library():
    r = library_report({"roots": [_root(0, 50, 0), _root(9, 10, 1), _root(9, 10, 1)]}, SPEC)
    assert r == {"verdict": "supported", "share": 0.1, "used": 2, "leaves": 20}


def test_no_library_leaf_refutes_and_a_small_share_is_inconclusive():
    assert library_report({"roots": [_root(9, 10, 0)]}, SPEC)["verdict"] == "refuted"
    assert library_report({"roots": [_root(9, 100, 3)]}, SPEC)["verdict"] == "inconclusive"


def test_without_a_library_there_is_nothing_to_judge():
    assert library_report({"roots": [_root(0, 10, 0)]}, SPEC) == {"verdict": "not_run"}
    assert library_report({"roots": [_root(5, 0, 0)]}, SPEC) == {"verdict": "not_run"}
