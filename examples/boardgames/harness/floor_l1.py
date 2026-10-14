"""§3.6 L1 — the leaf language, first step: fewer levels. Maps are now most of a strategy's bits (h249, h254), each
one bit per empty cell plus `level_bits` per levelled cell. L1 builds W6's 32 frontier positions with the full
pipeline exactly as W8 (built, improved, improved with lookahead) but with maps of 4 levels at 2 bits a levelled cell
instead of 8 at 3 — cheaper maps, perhaps more of them or more exceptions. Run-to-run variance is large (h243, h254),
so the bar is set against both 8-level runs of the same positions, W7 (69,847 bits) and W8 (74,714): supported at no
more than four fifths of the smaller, refuted at the larger or more (an unfinished root counts at its partial size,
so it can only refute); completion judged as W6."""
from __future__ import annotations

from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.floor_w8 import SPEC as W8

SPEC = {**{k: v for k, v in W8.items() if k not in ("w7_bits", "below_w7_support", "lookahead_support",
                                                    "lookahead_refute")},
        "builder": {**W8["builder"], "n_levels": 4, "level_bits": 2},
        "eight_level_bits": [69_847, 74_714],
        "support": 0.8}

l1_report = w6_report


def l1_size(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    bits = sum(r["bits"]["nodes"] for r in e["roots"])
    lower = not all(r["complete"] for r in e["roots"])
    verdict = ("refuted" if bits >= max(spec["eight_level_bits"])
               else "supported" if not lower and bits <= spec["support"] * min(spec["eight_level_bits"])
               else "inconclusive")
    return {"verdict": verdict, "bits": bits, "lower_bound": lower}
