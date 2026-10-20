"""§3.6 D1 — which zero-bit default to give the leaf rule next. Under L5's rule the safe moves a map leaves unlevelled
are played lowest first; a different order costs no bits, and a default that keeps the win more often should need
fewer levels and exceptions. D1 samples own positions with two or more safe moves and no winning move from L5's
certified leaves (scripts/c4_default_accuracy.py, 150 a root, 400 walked a leaf, seed 0), solves each exactly, and
scores each order's pick among all the safe moves by whether it keeps the win. The order to try: "tactics" — the
lowest after preferring moves that win by force in two, then moves that do not lose by force in two. Judged on its
misses against the lowest's: supported at two thirds of them or fewer, refuted above; undecidable when the lowest
misses fewer than 30 positions."""
from __future__ import annotations

from fractions import Fraction

SPEC = {"measurement_fp": "576f1422643b", "spec": "harness.floor_l5", "targets": "c49_l5.json.gz", "per_root": 150,
        "per_leaf": 400, "seed": 0, "order": "tactics", "support": Fraction(2, 3), "min_misses": 30}


def d1_misses(e: dict, spec: dict) -> dict:
    if any(e.get(k) != spec[k] for k in ("spec", "targets", "per_root", "per_leaf", "seed")) or (
            e.get("measurement_fingerprint") != spec["measurement_fp"]):
        return {"verdict": "not_run"}
    t = e["totals"]
    lowest = t["solved"] - t["hits"]["lowest"]
    order = t["solved"] - t["hits"][spec["order"]]
    if lowest < spec["min_misses"]:
        return {"verdict": "inconclusive", "lowest_misses": lowest, "misses": order}
    ratio = Fraction(order, lowest)
    return {"verdict": "supported" if ratio <= spec["support"] else "refuted", "lowest_misses": lowest,
            "misses": order, "ratio": round(float(ratio), 6)}
