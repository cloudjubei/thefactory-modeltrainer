"""§3.6 W5b — W5 rerun after the fix its overrun called for (h220): every walk the builder makes now stops at the
build's deadline, and the builder checks maps with the C verify (harness.native_leaf.verify). Otherwise W5 exactly:
the root-only long search ([[0, 1200], [2, 30]]), the C walk, W1's 8 positions, two library waves, a 2-hour cap;
judged as W5 (harness.floor_w5)."""
from __future__ import annotations

from harness.floor_w5 import SPEC as W5, time_report, w5_library, w5_report

SPEC = {**W5, "measurement_fp": "81a037019fab"}

w5b_report = w5_report
w5b_time = time_report
w5b_library = w5_library
