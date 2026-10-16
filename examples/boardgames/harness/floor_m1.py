"""§3.6 M1 — the leaf language, by its code: maps are 292,054 of F1's 538,731 position bits, each coded as one bit per
empty cell and three per levelled cell. harness.map_coding names exact cheaper codes (which cells are levelled, chosen
among the empty ones; the used levels by rank, or by how many cells each holds), none changing a move. M1 codes each of
F1's 3,196 maps with the cheapest of them plus a selector (scripts/c4_map_coding.py) — first reproducing F1's own map
bits exactly — and is judged on the share it saves: supported at a tenth or more, refuted under a twentieth."""
from __future__ import annotations

from fractions import Fraction

SPEC = {"measurement_fp": "133c5f0ac4b4", "source": "c49_f1.json.gz", "maps": 3_196, "f1_map_bits": 292_054,
        "n_levels": 8, "support": Fraction(1, 10), "refute": Fraction(1, 20)}


def m1_saving(e: dict, spec: dict) -> dict:
    if (e.get("measurement_fingerprint") != spec["measurement_fp"] or e.get("source") != spec["source"]
            or e.get("maps") != spec["maps"] or e["totals"]["bitmap"] != spec["f1_map_bits"]):
        return {"verdict": "not_run"}
    bits = e["totals"]["best"]
    saving = 1 - Fraction(bits, spec["f1_map_bits"])
    verdict = ("supported" if saving >= spec["support"] else "refuted" if saving < spec["refute"]
               else "inconclusive")
    return {"verdict": verdict, "saving": round(float(saving), 6), "bits": bits}
