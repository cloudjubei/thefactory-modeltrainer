"""Direct tests for harness/floor_s3.py — S3, complete certified strategies from ply-8 roots, judged exactly as S2
(harness.floor_s2.s2_report) under its own registered settings."""
from __future__ import annotations

from harness.floor_s2 import s2_report
from harness.floor_s3 import SPEC, s3_report


def test_s3_is_judged_by_the_s2_report_under_its_own_spec():
    assert s3_report is s2_report


def test_the_registered_study_builds_4_ply_8_roots_from_the_label_cache_with_leaves_from_ply_12():
    assert (SPEC["ply"], SPEC["roots"], SPEC["seed"], SPEC["source"]) == (8, 4, 3, "labels")
    assert SPEC["positions_fp"] == "bf548bd7612c"
    assert SPEC["builder"] == {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 4,
                               "reuse_window": 200, "seconds": 14400.0}
    assert SPEC["search"] == {"max_constraints": 20_000, "conflicts": 1_000_000, "seconds": 30.0, "cap": 1_000_000,
                              "lines": 64}
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (3, 1)
    assert (SPEC["compression_support"], SPEC["compression_refute"]) == (10, 1)
