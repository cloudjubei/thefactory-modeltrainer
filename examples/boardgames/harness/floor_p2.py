"""§3.6 P2 — P1 with the long search only at the frontier root: W1's three largest frontier strategies rebuilt with a
20-minute search for a pure steady state at the ply-8 position itself and W1's 30 seconds everywhere below, so a root
map is tried without the long searches at ply 10 that left P1's #316 and #230 with a quarter of W1's searches (h196).
The builder now caps every search at the build's remaining time. Judged as P1 (harness.floor_p1.p1_report) against W1
on the same positions."""
from __future__ import annotations

from harness.floor_p1 import SPEC as P1, p1_report

SPEC = {**P1, "measurement_fp": "2e4bc3465b9c",
        "builder": {**P1["builder"], "budgets": [[0, 1200.0]]}}

p2_report = p1_report
