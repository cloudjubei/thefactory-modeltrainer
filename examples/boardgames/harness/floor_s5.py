"""§3.6 S5 — earlier leaves: S4's study (size-scored local-search leaves over a shared oracle cache) with leaves searched
from ply 10 — two plies below the ply-8 roots instead of
four — and exceptions charged in walk order (harness.exception_coding, h183) both in the builder's accept test and in
the strategy's size. A leaf is kept with exceptions only at >= 30x under its table — what S4's ply-12 strategies
reach in walk order (31.7-34.7x, h183); at 10x the pilot's strategy settled at 10.5x (h185). Same 4 ply-8 roots of the label cache (seed 3) and the same 4-hour cap as S3/S4. Judged exactly
as S2-S4 (harness.floor_s2.s2_report)."""
from __future__ import annotations

from harness.floor_s2 import s2_report

SPEC = {
    "positions_fp": "bf548bd7612c",
    "measurement_fp": "605e04961a21",
    "ply": 8,
    "roots": 4,
    "seed": 3,
    "source": "labels",
    "builder": {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 2, "reuse_window": 200,
                "seconds": 14400.0, "accept": 30},
    "search": {"kind": "local", "seconds": 30.0, "cap": 1_000_000, "cache_limit": 1_500_000},
    "complete_support": 3,
    "complete_refute": 1,
    "compression_support": 10,
    "compression_refute": 1,
}

s5_report = s2_report
