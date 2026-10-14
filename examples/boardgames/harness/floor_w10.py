"""§3.6 W10 — maps shared across the frontier. Under L2's rule a third of W9's leaves are won from by a map L2 found on
other positions (h266), where under the exact rule a library went unused (h203). W10 rebuilds W9's 32 fresh positions
exactly as W9 (the same rule; the code now carries two-ply
safety, off) but starting every build from a library of the 90 maps L2 found (8,618 bits, paid once for the whole
game; a build pays only its references). Judged on completion as W6; on the whole game's projection — the opening,
the library once, and 671 x the 32 strategies' mean — against W9's 271,740 bits: at most four fifths supports, W9's or
more refutes (an unfinished root counts at its partial size, so it can only refute); and on the share of leaves that
use a library map: a fifth or more supports, under a twentieth refutes."""
from __future__ import annotations

from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.floor_w9 import SPEC as W9

SPEC = {**W9, "measurement_fp": "081011f9bbfa", "library_from": "c49_l2.json.gz", "library_bits": 8_618, "w9_bits": 271_740, "support": 0.8,
        "library_support": 0.2, "library_refute": 0.05}

w10_report = w6_report


def w10_projection(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    roots, opening = e["roots"], e["opening"]
    bits = (opening["moves"] * 4 + opening["trivial"] + spec["library_bits"]
            + opening["frontier"] * sum(r["bits"]["nodes"] for r in roots) / len(roots))
    ratio = bits / spec["w9_bits"]
    lower = not all(r["complete"] for r in roots)
    verdict = ("refuted" if ratio >= 1 else "supported" if not lower and ratio <= spec["support"] else "inconclusive")
    return {"verdict": verdict, "bits": bits, "ratio": ratio, "lower_bound": lower}


def w10_library(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    leaves = sum(r["bits"]["leaves"] for r in e["roots"])
    used = sum(r["library_leaves"] for r in e["roots"])
    share = used / leaves if leaves else 0.0
    verdict = ("supported" if share >= spec["library_support"] else "refuted" if share < spec["library_refute"]
               else "inconclusive")
    return {"verdict": verdict, "share": share, "used": used, "leaves": leaves}
