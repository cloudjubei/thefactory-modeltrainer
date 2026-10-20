"""§3.6 A6 — what the builder accepts, on the heavy tail. B1 found the heavy strategies' limit is the leaf-acceptance
rule: a leaf with exceptions is kept whenever it covers 30 times its bits in positions, and with more search time
searches near the root clear that bar with thousands of exceptions (h293). A6 raises the bar to 100x — the 30-s
searches, root budget and 2-hour cap unchanged — on B1's positions (F2's 16 heaviest), L5's rule and pipeline, no
library, and is judged on completion out of 16 (14 supports, 12 or fewer refutes) and on its total against B1's
control (harness.floor_b1_control: L5 as it stands, the same positions, judged under its own registration): supported at
nine tenths of it or less, refuted at the control's or more. A1 raised the bar under the exact rule and found it cut
both ways (h232)."""
from __future__ import annotations

from harness.floor_b1 import SPEC as B1
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_l4 import l4_size
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report

SPEC = {**B1, "builder": {**L5["builder"], "accept": 100}}

a6_report = w6_report


def a6_size(e: dict, control: dict) -> dict:
    return l4_size(e, control, SPEC, CONTROL)
