"""Direct tests for harness/floor_w5.py — W5, W2 with the root-only budget P2 meant (closed: [[0, 1200], [2, 30]]) and
the C walk, judged on completion and projection as W1 and on total build time against W2's."""
from __future__ import annotations

from harness.floor_w1 import w1_report
from harness.floor_w2 import SPEC as W2, library_report
from harness.floor_w5 import SPEC, time_report, w5_library, w5_report


def test_w5_closes_the_root_budget_and_walks_in_c_otherwise_as_w2():
    assert SPEC["builder"] == {**W2["builder"], "budgets": [[0, 1200.0], [2, 30.0]]}
    assert SPEC["search"] == {**W2["search"], "walker": "native"}
    assert SPEC["indices"] == W2["indices"] and SPEC["wave_size"] == W2["wave_size"]
    assert (SPEC["projection_support"], SPEC["projection_refute"]) == (1_618_856, 1_618_856)
    assert w5_report is w1_report and w5_library is library_report


def test_the_budget_closes_after_the_root():
    from harness.strategy_builder import Builder

    b = Builder.__new__(Builder)
    b.budgets, b.deadline = sorted(SPEC["builder"]["budgets"]), float("inf")
    assert b._budget(0) == 1200.0 and b._budget(2) == 30.0 and b._budget(8) == 30.0


def _roots(*seconds):
    return {"roots": [{"build_seconds": s} for s in seconds]}


def test_build_time_is_judged_against_w2_s_total():
    assert time_report(_roots(8_000.0, 8_116.0), SPEC) == {"verdict": "supported", "seconds": 16_116.0,
                                                           "ratio": 0.5}
    assert time_report(_roots(20_000.0), SPEC)["verdict"] == "inconclusive"
    assert time_report(_roots(32_232.0), SPEC)["verdict"] == "refuted"
