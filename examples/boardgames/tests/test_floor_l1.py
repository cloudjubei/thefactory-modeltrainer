"""Direct tests for harness/floor_l1.py — L1, W6's 32 frontier positions built with the full pipeline as W8 but with
maps of 4 levels (2 bits a levelled cell) instead of 8 (3 bits), judged on completion and on the 32 strategies' total
against the two 8-level runs of the same positions, W7 (69,847 bits) and W8 (74,714)."""
from __future__ import annotations

from harness.floor_l1 import SPEC, l1_report, l1_size
from harness.floor_w6 import w6_report
from harness.floor_w8 import SPEC as W8


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(SPEC["indices"], sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_l1_is_w8_with_four_levels_of_two_bits():
    assert SPEC["indices"] == W8["indices"] and SPEC["roots"] == 32
    assert SPEC["builder"] == {**W8["builder"], "n_levels": 4, "level_bits": 2}
    assert W8["builder"]["n_levels"] == 8 and W8["builder"]["level_bits"] == 3
    assert SPEC["search"] == W8["search"] and SPEC["measurement_fp"] == W8["measurement_fp"]
    assert (SPEC["eight_level_bits"], SPEC["support"]) == ([69_847, 74_714], 0.8)
    assert l1_report is w6_report


def test_the_size_is_judged_against_both_eight_level_runs():
    bar = 0.8 * 69_847
    r = l1_size(_e([int(bar)] + [0] * 31), SPEC)
    assert r == {"verdict": "supported", "bits": int(bar), "lower_bound": False}
    assert l1_size(_e([int(bar) + 1] + [0] * 31), SPEC)["verdict"] == "inconclusive"
    assert l1_size(_e([74_713] + [0] * 31), SPEC)["verdict"] == "inconclusive"
    assert l1_size(_e([74_714] + [0] * 31), SPEC)["verdict"] == "refuted"
    assert l1_size(_e([1_000] * 32), SPEC)["bits"] == 32_000
    exact = {**SPEC, "eight_level_bits": [100, 125]}
    assert l1_size(_e([80] + [0] * 31), exact)["verdict"] == "supported"
    assert l1_size(_e([125] + [0] * 31), exact)["verdict"] == "refuted"


def test_an_unfinished_root_can_only_refute():
    assert l1_size(_e([10] * 32, incomplete={471}), SPEC)["verdict"] == "inconclusive"
    assert l1_size(_e([74_714] + [0] * 31, incomplete={471}), SPEC)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([10] * 32, measurement_fingerprint="other"), _e([10] * 31), _e([10] * 32, builder=W8["builder"])):
        assert l1_size(bad, SPEC)["verdict"] == "not_run"
