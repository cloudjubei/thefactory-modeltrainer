"""§3.6 S3 — complete certified first-player strategies (table moves + steady-state leaves, harness.strategy_builder)
from ply-8 positions, one first-player move above S2's ply-10 roots, leaves still searched only from ply 12 (four
plies below the root). Roots: winning first-player positions at ply 8 of the label cache (books/c4_labels.json.gz,
identified by its file hash), the first `roots` non-trivial ones in a seeded shuffle. Judged exactly as S2: completion
within the cap and median compression against the same strategy as a 3-bit table (harness.floor_s2.s2_report). The
point is a measured basis for projecting a whole-game strategy's size while leaves start at ply 12."""
from __future__ import annotations

from harness.floor_s2 import s2_report

SPEC = {
    "positions_fp": "bf548bd7612c",
    "measurement_fp": "bfcc93d79ed7",
    "ply": 8,
    "roots": 4,
    "seed": 3,
    "source": "labels",
    "builder": {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 4, "reuse_window": 200,
                "seconds": 14400.0},
    "search": {"max_constraints": 20_000, "conflicts": 1_000_000, "seconds": 30.0, "cap": 1_000_000, "lines": 64},
    "complete_support": 3,
    "complete_refute": 1,
    "compression_support": 10,
    "compression_refute": 1,
}

s3_report = s2_report
