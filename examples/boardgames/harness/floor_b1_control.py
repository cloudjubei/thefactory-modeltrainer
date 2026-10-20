"""§3.6 B1's control — harness.floor_b1 with L5's budgets (30 s below the root): the same 16 positions, rule and
pipeline, no library, built right after the arm, so the longer searches are judged against a run of their own
conditions rather than F2's (built in waves from a growing library)."""
from __future__ import annotations

from harness.floor_b1 import SPEC as B1
from harness.floor_l5 import SPEC as L5

SPEC = {**B1, "builder": L5["builder"]}
