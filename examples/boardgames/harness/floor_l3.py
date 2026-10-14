"""§3.6 L3 — the leaf language, step 3: a deeper notion of safe. Under L2's rule (the first level holding any safe
move decides) a move is safe when the opponent cannot win at once after it. With `safe_depth` 2
(harness.steady_state.Facts) it is also unsafe when the opponent can answer it with a move after which every reply
loses at once — a forced win in two; when every 1-ply safe move fails that, they are kept. More positions are then
settled by the rule alone (one safe move), at no cost in bits. L3 builds W6's 32 frontier positions exactly as L2 with
only that changed. Judged on completion as W6, and on the 32 strategies' total against L2's 14,666 bits on the same
positions: supported at four fifths of it or less, refuted at L2's or more (an unfinished root counts at its partial
size, so it can only refute)."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report

SPEC = {**{k: v for k, v in L2.items() if k not in ("eight_level_bits",)},
        "measurement_fp": "081011f9bbfa",
        "builder": {**L2["builder"], "safe_depth": 2},
        "l2_bits": 14_666,
        "support": 0.8}

l3_report = w6_report


def l3_size(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    bits = sum(r["bits"]["nodes"] for r in e["roots"])
    ratio = bits / spec["l2_bits"]
    lower = not all(r["complete"] for r in e["roots"])
    verdict = ("refuted" if ratio >= 1 else "supported" if not lower and ratio <= spec["support"] else "inconclusive")
    return {"verdict": verdict, "bits": bits, "ratio": ratio, "lower_bound": lower}
