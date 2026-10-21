"""§3.6 K1 — richer leaves, paid in bits: a map with two phases. The heavy tail did not shrink under longer searches
(B1), a stricter acceptance bar (A6) or a better tie-break (L6, h301): its positions split into hundreds of leaves, each
with its own map. With `phases` a leaf's map may switch to second levels from a number of stones on
(harness.steady_state SWITCH), paying a switch, a second bitmap and its levels, and every map a flag bit; the local
search may switch one on, move or drop it. K1 builds L4's 32 positions with L6's rule and pipeline, no library, only
that changed, and is judged on completion as W6, on its total against L6 (harness.floor_l6, already built, under its
own registration) — supported at nine tenths of it or less, refuted at L6's or more — and on F1's 16 heaviest of the
32 alone against L6's same 16, the same way (an unfinished root counting at its partial size)."""
from __future__ import annotations

from harness.floor_l4 import SPEC as L4, l4_size
from harness.floor_l6 import SPEC as L6
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report

SPEC = {**L6, "measurement_fp": "98b7fb6ebf7f", "builder": {**L6["builder"], "phases": True},
        "heavy": L4["indices"][:16]}

k1_report = w6_report


def k1_size(e: dict, control: dict, spec: dict) -> dict:
    return l4_size(e, control, spec, L6)


def k1_heavy(e: dict, control: dict, spec: dict) -> dict:
    if _integrity(e, spec) or _integrity(control, L6):
        return {"verdict": "not_run"}
    heavy = set(spec["heavy"])
    arm = [r for r in e["roots"] if r["index"] in heavy]
    base = [r for r in control["roots"] if r["index"] in heavy]
    bits = sum(r["bits"]["nodes"] for r in arm)
    control_bits = sum(r["bits"]["nodes"] for r in base)
    ratio = bits / control_bits
    lower = not all(r["complete"] for r in arm)
    control_lower = not all(r["complete"] for r in base)
    verdict = ("refuted" if not control_lower and ratio >= 1 else
               "supported" if not lower and ratio <= spec["support"] else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control_bits, "ratio": ratio, "lower_bound": lower,
            "control_lower_bound": control_lower}
