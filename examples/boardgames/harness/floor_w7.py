"""§3.6 W7 — the whole game re-projected after the builder learned to improve anytime (h237, h239: four heavy
positions to 23% of W6's bits). W7 rebuilds W6's 32 frontier positions exactly as W6 built them plus the builder's
`improve` — a paired comparison on the same sample. Judged as W6 (harness.floor_w6): completion, and whether the whole
game still projects above the trained net's ~660K bits; and against W6's own projection of 4,306,421 bits — halved or
better supports, no smaller refutes (an unfinished root counts at its partial size, so it can only refute)."""
from __future__ import annotations

from harness.floor_w1 import _integrity
from harness.floor_w6 import SPEC as W6, w6_projection, w6_report

SPEC = {**W6, "measurement_fp": "79d96bd72825", "builder": {**W6["builder"], "improve": True},
        "w6_bits": 4_306_421, "halving_support": 0.5}

w7_report = w6_report
w7_projection = w6_projection


def w7_halving(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    roots, opening = e["roots"], e["opening"]
    bits = (opening["moves"] * 4 + opening["trivial"]
            + opening["frontier"] * sum(r["bits"]["nodes"] for r in roots) / len(roots))
    ratio = bits / spec["w6_bits"]
    lower = not all(r["complete"] for r in roots)
    verdict = ("refuted" if ratio >= 1 else "supported" if not lower and ratio <= spec["halving_support"]
               else "inconclusive")
    return {"verdict": verdict, "bits": bits, "ratio": ratio, "lower_bound": lower}
