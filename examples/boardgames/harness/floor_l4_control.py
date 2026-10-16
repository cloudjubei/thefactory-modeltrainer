"""§3.6 L4's control — harness.floor_l4 without `forcing`: the same 32 positions, L2's rule and pipeline, no library,
built the same way, so the forcing arm is judged against a run of its own conditions rather than F1's (built from a
growing library, on another day)."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_l4 import SPEC as L4

SPEC = {**L4, "builder": L2["builder"]}
