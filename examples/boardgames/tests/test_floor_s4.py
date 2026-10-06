"""Direct tests for harness/floor_s4.py — S4, S3's study with size-scored local-search leaves, judged exactly as S2
(harness.floor_s2.s2_report) under its own registered settings."""
from __future__ import annotations

from harness.floor_s2 import s2_report
from harness.floor_s3 import SPEC as S3
from harness.floor_s4 import SPEC, s4_report


def test_s4_is_judged_by_the_s2_report_under_its_own_spec():
    assert s4_report is s2_report


def test_s4_changes_only_the_leaf_search_from_s3():
    same = ("positions_fp", "ply", "roots", "seed", "source", "complete_support", "complete_refute",
            "compression_support", "compression_refute")
    assert {k: SPEC[k] for k in same} == {k: S3[k] for k in same}
    assert SPEC["builder"] == {**S3["builder"], "accept": 10}
    assert SPEC["search"] == {"kind": "local", "seconds": 30.0, "cap": 1_000_000, "cache_limit": 1_500_000}
    assert SPEC["measurement_fp"] == "7dd17b1d62ef"
