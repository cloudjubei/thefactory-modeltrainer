"""§3.6 A2 — choosing by size instead of by threshold. A1 showed a fixed threshold for keeping exceptions cuts both
ways (h232): splitting more shrank three heavy positions and made #565 4.7x larger. With the builder's
`choose_by_size`, a leaf with exceptions that clears W6's 30x is kept only where the split below it (built under the
same rule) would not make the strategy smaller. A2 rebuilds A1's four positions so, otherwise exactly as W6. Judged on
completion (all four) and on their total size: supported at no more than the smaller build per position of W6 and A1
(28,248 bits), refuted at A1's 42,464 or more (an unfinished root counts at its partial size, so it can only
refute)."""
from __future__ import annotations

from harness.floor_a1 import SPEC as A1
from harness.floor_w1 import _integrity
from harness.floor_w6 import SPEC as W6

SPEC = {**{k: v for k, v in A1.items() if k not in ("w6_bits", "size_support")},
        "measurement_fp": "a416e9353561",
        "builder": {**W6["builder"], "choose_by_size": True},
        "best_bits": 28_248,
        "a1_bits": 42_464}


def a2_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if not integrity and sorted(r["index"] for r in e["roots"]) != sorted(spec["indices"]):
        integrity = ["the roots are not the registered positions"]
    if integrity:
        return {"integrity": integrity, "complete": {"verdict": "not_run"}, "size": {"verdict": "not_run"}}
    roots = e["roots"]
    n = sum(1 for r in roots if r["complete"])
    complete = "supported" if n == len(roots) else "refuted" if n <= len(roots) // 2 else "inconclusive"
    bits = sum(r["bits"]["nodes"] for r in roots)
    lower = n < len(roots)
    size = ("refuted" if bits >= spec["a1_bits"] else "supported" if not lower and bits <= spec["best_bits"]
            else "inconclusive")
    return {"integrity": [], "complete": {"verdict": complete, "count": n},
            "size": {"verdict": size, "bits": bits, "lower_bound": lower}}
