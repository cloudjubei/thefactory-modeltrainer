"""§3.6 N1 — a learned default, paid once, piloted (user's choice). L6 broke ties by tactics, missing the win at 200
of D1's 4,523 positions against the lowest's 308, and the heavy tail barely moved (h301): a default must be right far
more often to replace levels and exceptions. N1 trains a small net (harness.learned_default: the cells as the mover
sees them and each move's safe / wins-in-two / loses-in-two facts, 32 hidden units, weights at 4 bits — 19,996 bits
with a scale a tensor) on positions drawn as D1's from F2's strategies for every root but L4's 32 (any the test holds
dropped), solved exactly, and scores its pick among the safe moves on D1's own positions
(scripts/c4_learned_default.py, which must reproduce D1's lowest and tactics hits). Judged on the quantized net's
misses: supported at half the tactics order's 200 or fewer, refuted at 200 or more — no better than a zero-bit
order."""
from __future__ import annotations

from harness.floor_l4 import SPEC as L4

SPEC = {"measurement_fp": "99fa81b04d0a",
        "test": {"spec": "harness.floor_l5", "targets": "c49_l5.json.gz", "per_leaf": 400, "per_root": 150, "seed": 0},
        "train": {"spec": "harness.floor_f2", "targets": "c49_f2.json.gz", "per_leaf": 400, "per_root": 150, "seed": 1,
                  "skip": L4["indices"]},
        "net": {"hidden": 32, "weight_bits": 4, "epochs": 40, "batch": 256, "lr": 0.001, "seed": 0},
        "d1": {"solved": 4_523, "lowest": 4_215, "tactics": 4_323}, "max_bits": 20_000, "support": 100, "refute": 200}


def n1_misses(e: dict) -> dict:
    t = e["totals"]
    if (e.get("measurement_fingerprint") != SPEC["measurement_fp"] or any(e.get(k) != SPEC[k] for k in
                                                                           ("test", "train", "net"))
            or any(t[k] != v for k, v in SPEC["d1"].items()) or t["bits"] > SPEC["max_bits"]):
        return {"verdict": "not_run"}
    misses = t["solved"] - t["net"]
    verdict = "supported" if misses <= SPEC["support"] else "refuted" if misses >= SPEC["refute"] else "inconclusive"
    return {"verdict": verdict, "misses": misses, "tactics_misses": t["solved"] - t["tactics"], "bits": t["bits"]}
