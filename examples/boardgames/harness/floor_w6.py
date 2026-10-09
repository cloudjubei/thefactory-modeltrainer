"""§3.6 W6 — how the whole game's size is spread over the frontier. W5b built W1's 8 sampled frontier positions
completely, and one of them (#230) held 73% of their bits (h224), so an 8-root mean says little. W6 builds the next 32
positions of W1's seeded order exactly as W5b built its 8 — the root-only long search, the C walk, a 2-hour cap — but
without a library, which W2 and W5b used for 2 of 61 and 0 of their leaves. Judged on the 32 alone: completion; whether
the whole game projects above the trained net (~660K bits) — W1's projection with a seeded bootstrap interval of the
sample's mean, above only when the whole interval is, below only when the whole interval is and every root finished
(an unfinished root counts at its partial size, a lower bound); and the share of the sample's bits its 4 largest
strategies carry, judged only when all 32 finished."""
from __future__ import annotations

import random
import statistics

from harness.floor_w1 import _integrity
from harness.floor_w5b import SPEC as W5B

SPEC = {**{k: v for k, v in W5B.items() if k not in ("projection_support", "projection_refute", "w2_build_seconds",
                                                     "time_support", "library_support", "wave_size")},
        "indices": [471, 178, 95, 486, 369, 506, 594, 324, 512, 534, 51, 374, 144, 413, 463, 251, 170, 667, 533, 197,
                    535, 403, 175, 98, 271, 53, 565, 428, 303, 639, 625, 457],
        "roots": 32,
        "complete_support": 28,
        "complete_refute": 24,
        "net_bits": 660_000,
        "bootstrap": 10_000,
        "bootstrap_seed": 6,
        "tail_count": 4,
        "tail_support": 0.5,
        "tail_refute": 0.25}


def w6_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if integrity:
        return {"integrity": integrity, "complete": {"verdict": "not_run"}}
    n = sum(1 for r in e["roots"] if r["complete"])
    verdict = ("supported" if n >= spec["complete_support"] else "refuted" if n <= spec["complete_refute"]
               else "inconclusive")
    return {"integrity": [], "complete": {"verdict": verdict, "count": n}}


def w6_projection(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    roots, opening = e["roots"], e["opening"]
    sizes = [r["bits"]["nodes"] for r in roots]
    rng = random.Random(spec["bootstrap_seed"])
    means = sorted(statistics.mean(rng.choices(sizes, k=len(sizes))) for _ in range(spec["bootstrap"]))
    base = opening["moves"] * 4 + opening["trivial"]
    bits, low, high = (base + opening["frontier"] * m
                       for m in (statistics.mean(sizes), means[int(0.025 * len(means))],
                                 means[int(0.975 * len(means)) - 1]))
    lower = not all(r["complete"] for r in roots)
    verdict = ("supported" if low > spec["net_bits"]
               else "refuted" if not lower and high < spec["net_bits"] else "inconclusive")
    hours = opening["frontier"] * statistics.mean(r["build_seconds"] for r in roots) / 3600
    return {"verdict": verdict, "bits": bits, "low": low, "high": high, "lower_bound": lower, "hours": hours}


def w6_tail(e: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    if not all(r["complete"] for r in e["roots"]):
        return {"verdict": "inconclusive"}
    sizes = sorted((r["bits"]["nodes"] for r in e["roots"]), reverse=True)
    largest = sizes[:spec["tail_count"]]
    share = sum(largest) / sum(sizes)
    verdict = ("supported" if share >= spec["tail_support"] else "refuted" if share < spec["tail_refute"]
               else "inconclusive")
    return {"verdict": verdict, "share": share, "largest": largest}
