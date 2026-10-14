"""Direct tests for harness/floor_a3.py — A3, A1's four heavy positions built exactly as W6 built them and then improved
for the rest of the 2 hours (the costliest leaves with exceptions split first, each split kept only if the strategy
shrinks), judged as A2 (harness.floor_a2.a2_report)."""
from __future__ import annotations

from harness.floor_a2 import SPEC as A2, a2_report
from harness.floor_a3 import SPEC, a3_report
from harness.floor_w6 import SPEC as W6


def test_a3_is_w6_on_a1_s_positions_then_improved_and_judged_as_a2():
    assert SPEC["indices"] == A2["indices"] == [95, 565, 594, 535] and SPEC["roots"] == 4
    assert SPEC["builder"] == {**W6["builder"], "improve": True}
    assert SPEC["search"] == W6["search"]
    assert SPEC["measurement_fp"] not in (W6["measurement_fp"], A2["measurement_fp"])
    assert (SPEC["best_bits"], SPEC["a1_bits"]) == (28_248, 42_464)
    assert a3_report is a2_report
