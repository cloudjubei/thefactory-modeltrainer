"""§3.6 R1 — the strategy's references, by their code: F2 charges every leaf the full width of an index into its
whole map list — the library shared with every earlier build, ~2,000 maps late in the game, and its own — 100,888 of
its 455,959 bits. harness.reference_coding's first_use names the library maps a build's leaves use once and codes
each leaf as a move-to-front rank in walk order, a build's own maps stored in the order of their first use so that
each first use costs one symbol; no move changes. Designed on F1's evidence only. R1 codes F2's 671 strategies so
(scripts/c4_reference_coding.py) — first reproducing F2's own reference bits exactly — and is judged on the share of
F2's whole strategy it saves: supported at a tenth or more, refuted under a twentieth."""
from __future__ import annotations

from fractions import Fraction

SPEC = {"measurement_fp": "f62cfb41901f", "source": "c49_f2.json.gz", "roots": 671, "f2_bits": 455_959,
        "f2_reference_bits": 100_888, "support": Fraction(1, 10), "refute": Fraction(1, 20)}


def r1_saving(e: dict, spec: dict) -> dict:
    if (e.get("measurement_fingerprint") != spec["measurement_fp"] or e.get("source") != spec["source"]
            or e.get("roots") != spec["roots"] or e["totals"]["fixed"] != spec["f2_reference_bits"]):
        return {"verdict": "not_run"}
    references = e["totals"]["first_use"]
    saving = Fraction(spec["f2_reference_bits"] - references, spec["f2_bits"])
    verdict = ("supported" if saving >= spec["support"] else "refuted" if saving < spec["refute"]
               else "inconclusive")
    return {"verdict": verdict, "saving": round(float(saving), 6),
            "bits": spec["f2_bits"] - spec["f2_reference_bits"] + references, "references": references}
