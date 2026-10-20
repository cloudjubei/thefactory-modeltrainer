"""§3.6 B1 — the search, on the heavy tail. F2 left the whole game in 455,959 bits, its 67 heaviest positions holding 57%
of them (h290); there the builder runs out of maps that win near the root and splits, each deeper leaf search getting
30 s. B1 gives those deeper searches 120 s — the 20-minute root search and the 2-hour cap unchanged — on F2's 16
heaviest positions, L5's rule and pipeline, no library, and a control built the same way with 30 s
(harness.floor_b1_control). Longer searches near the root starved the builds of P1-P2, so completion is judged as well
as size: each arm out of 16 (14 supports, 12 or fewer refutes), and the arm's total against the control's — supported at
nine tenths of it or less, refuted at the control's or more (unfinished roots counting at their partial size)."""
from __future__ import annotations

from harness.floor_l4 import l4_size
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report

SPEC = {**L5,
        "indices": [207, 113, 216, 240, 218, 467, 16, 622, 605, 1, 487, 584, 58, 541, 542, 212],
        "roots": 16, "complete_support": 14, "complete_refute": 12,
        "builder": {**L5["builder"], "budgets": [[0, 1200.0], [2, 120.0]]},
        "support": 0.9}

b1_report = w6_report


def b1_size(e: dict, control: dict) -> dict:
    from harness.floor_b1_control import SPEC as CONTROL

    return l4_size(e, control, SPEC, CONTROL)
