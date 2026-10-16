"""§3.6 L4 — the leaf language, step 4: prefer forcing moves. Under L2's rule (the first level holding any safe move
decides) every safe move is a candidate. With `forcing` (harness.steady_state.Facts), of several safe moves only those
that FORCE are kept when any do — those leaving the opponent at most one reply after which the mover cannot win at
once — so play keeps the opponent's choices to one wherever it can, at no cost in bits. F1 left the whole game in
548,424 bits, 58% of them in its heaviest tenth of positions (h274), and showed a random 32 positions misleads on such
a frontier (h276); so L4 builds 32 of F1's positions — its 16 heaviest and 16 spread evenly over the rest by F1's size —
with L2's rule and pipeline and no library, with `forcing` (this spec) and without it (harness.floor_l4_control), and
judges each arm's completion as W6 and the forcing arm's total against the control's: supported at four fifths of it
or less, refuted at the control's or more. An unfinished root counts at its partial size — a lower bound — so in the
forcing arm it can only refute, in the control only support."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report

SPEC = {**{k: v for k, v in L2.items() if k not in ("eight_level_bits",)},
        "measurement_fp": "1d12399d877c",
        "indices": [218, 207, 113, 240, 487, 598, 61, 216, 467, 541, 1, 297, 652, 584, 358, 637,
                    226, 530, 121, 15, 94, 324, 276, 548, 472, 449, 470, 568, 204, 159, 9, 353],
        "builder": {**L2["builder"], "forcing": True},
        "support": 0.8}

l4_report = w6_report


def l4_size(e: dict, control: dict, spec: dict, control_spec: dict) -> dict:
    if _integrity(e, spec) or _integrity(control, control_spec):
        return {"verdict": "not_run"}
    bits = sum(r["bits"]["nodes"] for r in e["roots"])
    control_bits = sum(r["bits"]["nodes"] for r in control["roots"])
    ratio = bits / control_bits
    lower = not all(r["complete"] for r in e["roots"])
    control_lower = not all(r["complete"] for r in control["roots"])
    verdict = ("refuted" if not control_lower and ratio >= 1 else
               "supported" if not lower and ratio <= spec["support"] else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control_bits, "ratio": ratio, "lower_bound": lower,
            "control_lower_bound": control_lower}
