"""§3.6 S4 — S3 again with one change: leaves searched by size-scored local search (harness.steady_local over a shared
bounded oracle cache) instead of SAT, so a leaf search never fails outright — a leaf may carry exceptions when it is
at least `accept` times smaller than its own positions as a 3-bit table (harness.strategy_builder). Same 4 ply-8
roots of the label cache (seed 3), same leaf depth (ply 12), same 4-hour cap. Judged exactly as S2/S3
(harness.floor_s2.s2_report); the comparison with S3 is on the same roots."""
from __future__ import annotations

from harness.floor_s2 import s2_report

SPEC = {
    "positions_fp": "bf548bd7612c",
    "measurement_fp": "7dd17b1d62ef",
    "ply": 8,
    "roots": 4,
    "seed": 3,
    "source": "labels",
    "builder": {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 4, "reuse_window": 200,
                "seconds": 14400.0, "accept": 10},
    "search": {"kind": "local", "seconds": 30.0, "cap": 1_000_000, "cache_limit": 1_500_000},
    "complete_support": 3,
    "complete_refute": 1,
    "compression_support": 10,
    "compression_refute": 1,
}

s4_report = s2_report
