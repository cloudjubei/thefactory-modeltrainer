"""§3.6 F3 — the whole game, rebuilt with L6's rule. L6 (harness.floor_l6) breaks a deciding level's ties by tactics:
on L4's 32 positions it took 0.94 of L5's bits (h300 inconclusive), the light positions 0.66 and F1's heaviest 0.97
(h301). F3 is F2 (harness.floor_f2) with only that changed, under today's code (K1's fingerprint; L5's rule plays as it
did): every one of the 671 frontier positions in W1's seeded order, waves of 32, each starting from a library of every
map found so far seeded with L2's 90 maps (8,618 bits, paid once), L2's pipeline. Judged on completion as F1, and on
the whole strategy's size — the opening, the seed library once, every position's own bits — against F2's 455,959:
supported at nine tenths of it or less, refuted at F2's or more (a partial total counting as a lower bound)."""
from __future__ import annotations

from harness.floor_f1 import f1_complete
from harness.floor_f2 import SPEC as F2, f2_size
from harness.floor_k1 import SPEC as K1
from harness.floor_l6 import SPEC as L6

SPEC = {**{k: v for k, v in F2.items() if k != "f1_bits"}, "measurement_fp": K1["measurement_fp"],
        "builder": L6["builder"], "f2_bits": 455_959}

f3_complete = f1_complete


def f3_size(e: dict, spec: dict) -> dict:
    return f2_size(e, {**spec, "f1_bits": spec["f2_bits"]})
