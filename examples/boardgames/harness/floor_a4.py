"""§3.6 A4 — improving with lookahead. In W7 four heavy positions kept no split at all: each was one leaf at the root
whose split one level down was larger, so the greedy improvement never looked below it (h243). With lookahead a split
that is not smaller at once has the leaves it made improved first, and is judged with them. A4 builds eight positions
— those four (#95, #271, #374, #512) and four heavy ones (#594, #535, #98, #457) — as W6, improves them as W7, and,
once that runs dry, improves them again with lookahead: a paired measurement inside each build, free of the run-to-run
variance between builds (h243). Judged on how much further lookahead cuts the total plain improvement left
(supported at >= 15%, refuted under 3%), and on the heavy positions plain improvement could not change (more than 200
bits, no split kept): supported when lookahead cuts at least half of them, refuted when it cuts none, undecided with
fewer than two. Either is undecided while a position is unfinished."""
from __future__ import annotations

from harness.floor_w1 import _integrity
from harness.floor_w6 import SPEC as W6

SPEC = {**{k: v for k, v in W6.items() if k not in ("complete_support", "complete_refute", "net_bits", "bootstrap",
                                                    "bootstrap_seed", "tail_count", "tail_support", "tail_refute")},
        "measurement_fp": "0594eb30c588",
        "indices": [95, 271, 374, 512, 594, 535, 98, 457],
        "roots": 8,
        "builder": {**W6["builder"], "improve": True, "lookahead": True},
        "further_support": 0.15,
        "further_refute": 0.03,
        "stuck_above": 200}


def _checked(e: dict, spec: dict) -> bool:
    return not _integrity(e, spec) and sorted(r["index"] for r in e["roots"]) == sorted(spec["indices"])


def a4_further(e: dict, spec: dict) -> dict:
    if not _checked(e, spec):
        return {"verdict": "not_run"}
    if not all(r["complete"] for r in e["roots"]):
        return {"verdict": "inconclusive"}
    plain = sum(r["improved"]["plain_bits"] for r in e["roots"])
    bits = sum(r["bits"]["nodes"] for r in e["roots"])
    cut = (plain - bits) / plain if plain else 0
    verdict = ("supported" if cut >= spec["further_support"] else "refuted" if cut < spec["further_refute"]
               else "inconclusive")
    return {"verdict": verdict, "plain_bits": plain, "bits": bits, "cut": cut}


def a4_unstuck(e: dict, spec: dict) -> dict:
    if not _checked(e, spec):
        return {"verdict": "not_run"}
    if not all(r["complete"] for r in e["roots"]):
        return {"verdict": "inconclusive"}
    stuck = [r for r in e["roots"] if r["improved"]["kept"] == 0 and r["improved"]["plain_bits"] > spec["stuck_above"]]
    cut = sum(1 for r in stuck if r["bits"]["nodes"] < r["improved"]["plain_bits"])
    verdict = ("inconclusive" if len(stuck) < 2 else "supported" if 2 * cut >= len(stuck) else "refuted" if cut == 0
               else "inconclusive")
    return {"verdict": verdict, "stuck": len(stuck), "cut": cut}
