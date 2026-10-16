"""§3.6 L5 — the leaf language, step 5: unlevelled cells as a last level. Under L2's rule a map leaves the rule without
a move wherever several safe moves stand and none is on a levelled cell — an exception must cover it, or the map must
level more cells. With `unlevelled` "last" (harness.steady_state.Facts) the safe moves on unlevelled cells form one
more level after the map's, read like the others (the lowest decides), so such positions get a default move and a map
can drop every level that only reproduces it — at no cost in bits. L5 builds L4's 32 positions (F1's 16 heaviest and
16 spread over the rest) with L2's rule and pipeline, no library, only that changed, and is judged on completion as W6
and on its total against L4's control (harness.floor_l4_control: the same positions, L2's rule as it stands, no library,
judged under its own registration): supported at four fifths of it or less, refuted at the control's or more."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_l4 import SPEC as L4, l4_size
from harness.floor_l4_control import SPEC as CONTROL
from harness.floor_w6 import w6_report

SPEC = {**L4, "measurement_fp": "7717c79f94ea", "builder": {**L2["builder"], "unlevelled": "last"}}

l5_report = w6_report


def l5_size(e: dict, control: dict, spec: dict) -> dict:
    return l4_size(e, control, spec, CONTROL)
