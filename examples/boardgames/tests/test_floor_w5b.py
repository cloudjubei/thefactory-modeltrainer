"""Direct tests for harness/floor_w5b.py — W5b, W5 rerun after the deadline fix, judged exactly as W5."""
from __future__ import annotations

from harness.floor_w5 import SPEC as W5, time_report, w5_library, w5_report
from harness.floor_w5b import SPEC, w5b_library, w5b_report, w5b_time


def test_w5b_is_w5_with_its_own_fingerprint():
    assert {k: v for k, v in SPEC.items() if k != "measurement_fp"} == \
        {k: v for k, v in W5.items() if k != "measurement_fp"}
    assert SPEC["measurement_fp"] != W5["measurement_fp"]


def test_w5b_is_judged_as_w5():
    assert w5b_report is w5_report and w5b_time is time_report and w5b_library is w5_library
