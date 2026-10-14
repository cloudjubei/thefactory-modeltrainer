"""§3.6 L2 — the leaf language, step 2: a level decides with any safe move. Under the rule so far a level decides only
when it holds exactly one safe move, so a map leaves undefined every position where its first useful level holds
two — and those cost exceptions, or more maps (h249: maps are now most of the bits). With `ties` "lowest"
(harness.steady_state.Facts) the first level holding any safe move decides, with its lowest-numbered move: a map
defines play in more positions at no cost in bits. L2 builds W6's 32 frontier positions with W8's full pipeline and 8
levels at 3 bits, only the rule changed, judged as L1 (harness.floor_l1): completion, and the 32 strategies' total
against both runs of the exact rule — supported at no more than four fifths of W7's 69,847 bits, refuted at W8's
74,714 or more."""
from __future__ import annotations

from harness.floor_l1 import SPEC as L1, l1_report, l1_size
from harness.floor_w8 import SPEC as W8

SPEC = {**L1, "measurement_fp": "e5968103e6f4", "builder": {**W8["builder"], "ties": "lowest"}}

l2_report = l1_report
l2_size = l1_size
