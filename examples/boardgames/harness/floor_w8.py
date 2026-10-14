"""§3.6 W8 — the whole game with the full pipeline: W6's 32 frontier positions built as W6, improved, and improved
again with lookahead (h244: a further fifth on heavy positions). `research` is left out — it found almost nothing to
do (h247, h249). Judged as W6 on completion and against the trained net (~660K bits); against W7's projection of
1,465,679 bits — at most nine tenths supports, W7's or more refutes (an unfinished root counts at its partial size,
so it can only refute); and on lookahead's cut of what plain improvement left, measured inside the 32 builds
(supported at >= 10%, refuted under 2%, undecided while a position is unfinished)."""
from __future__ import annotations

from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_projection, w6_report
from harness.floor_w7 import SPEC as W7

SPEC = {**{k: v for k, v in W7.items() if k not in ("w6_bits", "halving_support")},
        "measurement_fp": "b2f5efcf2ac9",
        "builder": {**W7["builder"], "lookahead": True},
        "w7_bits": 1_465_679,
        "below_w7_support": 0.9,
        "lookahead_support": 0.10,
        "lookahead_refute": 0.02}

w8_report = w6_report
w8_projection = w6_projection


def w8_vs_w7(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    roots, opening = e["roots"], e["opening"]
    bits = (opening["moves"] * 4 + opening["trivial"]
            + opening["frontier"] * sum(r["bits"]["nodes"] for r in roots) / len(roots))
    ratio = bits / spec["w7_bits"]
    lower = not all(r["complete"] for r in roots)
    verdict = ("refuted" if ratio >= 1 else "supported" if not lower and ratio <= spec["below_w7_support"]
               else "inconclusive")
    return {"verdict": verdict, "bits": bits, "ratio": ratio, "lower_bound": lower}


def w8_lookahead(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    if not all(r["complete"] for r in e["roots"]):
        return {"verdict": "inconclusive"}
    plain = sum(r["improved"]["plain_bits"] for r in e["roots"])
    bits = sum(r["bits"]["nodes"] for r in e["roots"])
    cut = (plain - bits) / plain if plain else 0
    verdict = ("supported" if cut >= spec["lookahead_support"] else "refuted" if cut < spec["lookahead_refute"]
               else "inconclusive")
    return {"verdict": verdict, "plain_bits": plain, "bits": bits, "cut": cut}
