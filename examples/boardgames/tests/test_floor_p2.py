"""Direct tests for harness/floor_p2.py — P2, P1 with the long search only at the frontier root, judged as P1."""
from __future__ import annotations

from harness.floor_p1 import SPEC as P1, p1_report
from harness.floor_p2 import SPEC, p2_report


def test_p2_is_judged_by_the_p1_report():
    assert p2_report is p1_report


def test_p2_changes_only_the_budgets_from_p1():
    assert {k: v for k, v in SPEC.items() if k not in ("builder", "measurement_fp")} == \
        {k: v for k, v in P1.items() if k not in ("builder", "measurement_fp")}
    assert SPEC["builder"] == {**P1["builder"], "budgets": [[0, 1200.0]]}
    assert SPEC["measurement_fp"] != P1["measurement_fp"]
