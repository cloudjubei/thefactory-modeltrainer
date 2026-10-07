"""Direct tests for harness/floor_s5.py — S5, S4's study with leaves from ply 10 and walk-order exception accounting,
judged exactly as S2 (harness.floor_s2.s2_report) under its own registered settings."""
from __future__ import annotations

from harness.floor_s2 import s2_report
from harness.floor_s4 import SPEC as S4
from harness.floor_s5 import SPEC, s5_report


def test_s5_is_judged_by_the_s2_report_under_its_own_spec():
    assert s5_report is s2_report


def test_s5_changes_only_the_leaf_depth_and_the_acceptance_bar_from_s4():
    same = ("positions_fp", "ply", "roots", "seed", "source", "search", "complete_support", "complete_refute",
            "compression_support", "compression_refute")
    assert {k: SPEC[k] for k in same} == {k: S4[k] for k in same}
    assert SPEC["builder"] == {**S4["builder"], "min_leaf_depth": 2, "accept": 30}
    assert SPEC["measurement_fp"] == "605e04961a21"
