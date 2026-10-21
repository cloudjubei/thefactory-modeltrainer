"""§3.6 L6 — the leaf language, step 6: ties broken by tactics. Under L5's rule a deciding level holding several safe
moves — the unlevelled last level above all — plays the lowest. D1 (h297) found that on positions from L5's certified
leaves the lowest fails to keep the win at 308 of 4,523, and "tactics" — the lowest of the moves that win by force in
two, else of those that do not lose by force in two, else the lowest (harness.steady_state.Facts `order`) — at 200,
no bits added. L6 builds L4's 32 positions with L5's rule and pipeline, no library, only that changed, and is judged
on completion as W6 and on its total against L5 (harness.floor_l5: the same positions, already built, judged under
its own registration): supported at nine tenths of it or less, refuted at L5's or more."""
from __future__ import annotations

from harness.floor_l4 import l4_size
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report

SPEC = {**L5, "measurement_fp": "ce56e01571bb", "builder": {**L5["builder"], "order": "tactics"}, "support": 0.9}

l6_report = w6_report


def l6_size(e: dict, control: dict, spec: dict) -> dict:
    return l4_size(e, control, spec, L5)
