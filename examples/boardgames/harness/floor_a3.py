"""§3.6 A3 — choosing by size, anytime. A2's recursive comparison beat both fixed thresholds where it finished but
did not fit 2 hours (h235). A3 builds A1's four positions exactly as W6 built them — complete first — and then, with
the builder's `improve`, spends the rest of the 2 hours splitting the leaves with exceptions whose code costs most,
keeping each split only if the whole strategy shrinks. Judged as A2: all four complete, and their total against the
smaller build per position of W6 and A1 (28,248 bits; refuted at A1's 42,464 or more)."""
from __future__ import annotations

from harness.floor_a2 import SPEC as A2, a2_report
from harness.floor_w6 import SPEC as W6

SPEC = {**A2, "measurement_fp": "79d96bd72825", "builder": {**W6["builder"], "improve": True}}

a3_report = a2_report
