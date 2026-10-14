"""Direct tests for harness/floor_l2.py — L2, W6's 32 frontier positions built with W8's pipeline and 8 levels, the
rule's levels read with ties to the lowest move, judged as L1."""
from __future__ import annotations

from harness.floor_l1 import SPEC as L1, l1_report, l1_size
from harness.floor_l2 import SPEC, l2_report, l2_size
from harness.floor_w8 import SPEC as W8


def test_l2_is_w8_with_ties_to_the_lowest_move_judged_as_l1():
    assert SPEC["indices"] == W8["indices"] and SPEC["roots"] == 32
    assert SPEC["builder"] == {**W8["builder"], "ties": "lowest"}
    assert SPEC["builder"]["n_levels"] == 8 and SPEC["builder"]["level_bits"] == 3
    assert SPEC["search"] == W8["search"] and SPEC["measurement_fp"] == "e5968103e6f4" != W8["measurement_fp"]
    assert (SPEC["eight_level_bits"], SPEC["support"]) == (L1["eight_level_bits"], L1["support"]) == (
        [69_847, 74_714], 0.8)
    assert l2_report is l1_report and l2_size is l1_size
