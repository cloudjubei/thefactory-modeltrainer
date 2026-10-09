"""§3.6 A1 — does the builder keep exceptions it would do better to split? Exceptions are 84% of W6's bits (h228), and
the builder keeps the FIRST leaf with exceptions that is `accept` times under its own table (30x in W6) without asking
whether a move and smaller leaves below it would cost less. A1 rebuilds four of W6's heavy positions exactly as W6
built them — two that W6 left as one root map with exceptions (#95, #565), two it had already split (#594, #535) — with
`accept` raised to 100x, so more leaves are split. Judged on completion (all four) and on their total size against
W6's: smaller at <= 4/5 of it, refuted at W6's or more (an unfinished root counts at its partial size, so it can only
refute)."""
from __future__ import annotations

from harness.floor_w1 import _integrity
from harness.floor_w6 import SPEC as W6

SPEC = {**{k: v for k, v in W6.items() if k not in ("complete_support", "complete_refute", "net_bits", "bootstrap",
                                                    "bootstrap_seed", "tail_count", "tail_support", "tail_refute")},
        "indices": [95, 565, 594, 535],
        "roots": 4,
        "builder": {**W6["builder"], "accept": 100},
        "w6_bits": {95: 6453, 565: 3877, 594: 32544, 535: 30009},
        "size_support": 0.8}


def a1_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if not integrity and sorted(r["index"] for r in e["roots"]) != sorted(spec["w6_bits"]):
        integrity = ["the roots are not the registered positions"]
    if integrity:
        return {"integrity": integrity, "complete": {"verdict": "not_run"}, "size": {"verdict": "not_run"}}
    roots = e["roots"]
    n = sum(1 for r in roots if r["complete"])
    complete = "supported" if n == len(roots) else "refuted" if n <= len(roots) // 2 else "inconclusive"
    bits = sum(r["bits"]["nodes"] for r in roots)
    w6 = sum(spec["w6_bits"].values())
    ratio = bits / w6
    lower = n < len(roots)
    size = ("refuted" if ratio >= 1 else "supported" if not lower and ratio <= spec["size_support"]
            else "inconclusive")
    return {"integrity": [], "complete": {"verdict": complete, "count": n},
            "size": {"verdict": size, "bits": bits, "w6_bits": w6, "ratio": ratio, "lower_bound": lower}}
