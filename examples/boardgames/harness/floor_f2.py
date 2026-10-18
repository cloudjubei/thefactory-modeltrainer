"""§3.6 F2 — the whole game, rebuilt with L5's rule. L5 (harness.floor_l5) reads the safe moves on cells a map leaves
without a level as a last level: on 32 of F1's positions it took 0.84 of a same-rule control's bits, smaller on 24
(h287), its gain in the heavy tail where F1 holds 58% of its bits. F2 is F1 (harness.floor_f1) with only that changed:
every one of the 671 frontier positions in W1's seeded order, waves of 32, each starting from a library of every map
found so far seeded with L2's 90 maps (8,618 bits, paid once), L2's pipeline. Judged on completion as F1, and on the
whole strategy's size — the opening, the seed library once, every position's own bits — against F1's 548,424:
supported at nine tenths of it or less, refuted at F1's or more (a partial total counting as a lower bound)."""
from __future__ import annotations

from harness.floor_f1 import SPEC as F1, f1_bits, f1_checked, f1_complete
from harness.floor_l5 import SPEC as L5

SPEC = {**F1, "measurement_fp": L5["measurement_fp"], "builder": L5["builder"], "f1_bits": 548_424,
        "support": (9, 10)}

f2_complete = f1_complete


def f2_size(e: dict, spec: dict) -> dict:
    if not f1_checked(e, spec):
        return {"verdict": "not_run"}
    bits = f1_bits(e, spec)
    lower = not all(r["complete"] for r in e["roots"])
    num, den = spec["support"]
    verdict = ("refuted" if bits >= spec["f1_bits"] else
               "supported" if not lower and den * bits <= num * spec["f1_bits"] else "inconclusive")
    return {"verdict": verdict, "bits": bits, "lower_bound": lower}
