"""§3.6 Y1 — WeakC4 (2swap; ≈35-45K bits entropy-coded, ≈90K shipped) as a yardstick for G3's certified
280,640-bit whole game: is the gap the rule language, or the tree and its search? scripts/c4_weakc4_yardstick.py
walks WeakC4's strategy from the empty board and reads each leaf's diagram under our rule (harness.steady_state.Facts).
Judged on:

  - the language — the share of WeakC4's leaves whose diagram, translated cell for cell, wins at its position under
    our "exact" rule with all ten of its levels (its own reading but for our safe-move filter): supported at nine
    tenths or more (our language holds WeakC4's), refuted at half or less;
  - the price — WeakC4's whole solution in our code (a bit per node, three per table move, references by first_use,
    every map by map_bits), in the cheapest reading whose diagrams win at every leaf, against G3's bits: supported at a
    third or less (the gap is the tree and its search, not our code), refuted at half or more; not judged when no
    reading wins every leaf."""
from __future__ import annotations

SPEC = {"g3_bits": 280_640, "language_support": (9, 10), "language_refute": (1, 2), "price_support": (1, 3),
        "price_refute": (1, 2)}


def y1_language(e: dict, spec: dict) -> dict:
    r = e["readings"]["exact_all"]
    if not e.get("weakc4_commit") or not r["leaves"]:
        return {"verdict": "not_run"}
    (sn, sd), (rn, rd) = spec["language_support"], spec["language_refute"]
    won, leaves = r["won"], r["leaves"]
    verdict = "supported" if sd * won >= sn * leaves else "refuted" if rd * won <= rn * leaves else "inconclusive"
    return {"verdict": verdict, "won": won, "leaves": leaves}


def y1_price(e: dict, spec: dict) -> dict:
    whole = [(r["bits"]["total"], name) for name, r in e["readings"].items() if r["leaves"] and r["won"] == r["leaves"]]
    if not e.get("weakc4_commit") or not whole:
        return {"verdict": "not_run"}
    bits, reading = min(whole)
    (sn, sd), (rn, rd) = spec["price_support"], spec["price_refute"]
    g3 = spec["g3_bits"]
    verdict = "supported" if sd * bits <= sn * g3 else "refuted" if rd * bits >= rn * g3 else "inconclusive"
    return {"verdict": verdict, "bits": bits, "reading": reading, "ratio": bits / g3}
