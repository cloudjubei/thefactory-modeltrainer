"""§3.6 A7 — the other side of A6's bar. A6 kept leaves with exceptions only at 100x their bits in positions and the
heavy tail grew: the build split at once, without comparing, where improvement would have kept the leaf (h296). A7
lowers the bar to 10x — the 30-s searches, root budget and 2-hour cap unchanged — so the build keeps more such leaves
and the improvement pass, which splits a leaf only where the split is smaller, has more of them to choose from. On B1's
positions (F2's 16 heaviest), L5's rule and pipeline, no library, under today's code (K1's fingerprint: L5's rule plays
as it did, h301), judged on completion out of 16 (14 supports, 12 or fewer refutes) and on its total against B1's
control (harness.floor_b1_control, judged under its own registration): supported at nine tenths of it or less, refuted
at the control's or more. B1's longer searches let exception-heavy leaves in and grew the tail (h293)."""
from __future__ import annotations

from harness.floor_b1 import SPEC as B1
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_k1 import SPEC as K1
from harness.floor_l4 import l4_size
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report

SPEC = {**B1, "measurement_fp": K1["measurement_fp"], "builder": {**L5["builder"], "accept": 10}}

a7_report = w6_report


def a7_size(e: dict, control: dict) -> dict:
    return l4_size(e, control, SPEC, CONTROL)
