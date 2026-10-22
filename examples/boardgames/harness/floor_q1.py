"""§3.6 Q1 — exceptions, by their code: F2 spends 107,049 of its bits on exceptions coded in walk order
(harness.exception_coding) — which contested positions are flagged, an enumerative code, and each one's move among the
safe moves the map did not play. Q1 asks whether either part can be predicted for free: the flags coded per class
(the map's move from a levelled cell or the unlevelled last level, the tactics order agreeing with it or not —
harness.steady_state.Facts `order`), the moves tactics first (a bit for whether it is the tactics move, where that is
one of them). On a seeded sample of 60 of F2's leaves with exceptions (scripts/c4_exception_classes.py), first
reproducing their coded bits, each code is judged on the share of those bits it saves: supported at a tenth or more,
refuted at none. Piloted on L5's leaves, where every class held exceptions at about the same rate."""
from __future__ import annotations

from fractions import Fraction

SPEC = {"measurement_fp": "0183157a95c4", "source": "c49_f2.json.gz", "seed": 0, "leaves": 60,
        "support": Fraction(1, 10)}


def _checked(e: dict) -> bool:
    t = e["totals"]
    return (e.get("measurement_fingerprint") == SPEC["measurement_fp"] and e.get("source") == SPEC["source"]
            and e.get("seed") == SPEC["seed"] and e.get("leaves") == SPEC["leaves"]
            and t["one_class"] + t["plain"] == t["coded"])


def _judge(e: dict, before: str, after: str) -> dict:
    if not _checked(e):
        return {"verdict": "not_run"}
    t = e["totals"]
    saving = Fraction(t[before] - t[after], t["coded"])
    verdict = "supported" if saving >= SPEC["support"] else "refuted" if saving <= 0 else "inconclusive"
    return {"verdict": verdict, "saving": round(float(saving), 6)}


def q1_flags(e: dict) -> dict:
    return _judge(e, "one_class", "per_class")


def q1_moves(e: dict) -> dict:
    return _judge(e, "plain", "tactics_first")
