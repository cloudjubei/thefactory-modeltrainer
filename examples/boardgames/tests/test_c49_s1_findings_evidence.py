"""§3.6 S1 — findings registered AFTER the data, from evidence/c49_s1.json.gz (the run judged by h153, supported, and
h154, inconclusive)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_s1 import SPEC

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _rows() -> list:
    return load_evidence(EVIDENCE / FILE)["results"]


def _found(ply):
    return [r for r in _rows() if r["ply"] == ply and r["status"] == "found"]


def _compression(r):
    return 3 * r["own_positions"] / r["bits"]


def test_c49_s1_compression_is_bimodal_small_states_cost_about_a_table_large_ones_10_to_432_times_less():
    values = sorted(_compression(r) for r in _found(14))
    small, large = [v for v in values if v < 10], [v for v in values if v >= 10]
    assert len(small) == 14 and len(large) == 11 and max(small) < 8 and min(large) > 11
    assert max(values) > 430 and all(r["bits"] <= 94 for r in _found(14))
    assert max(r["own_positions"] for r in _found(14)) == 12229
    assert sum(1 for r in _found(12) if _compression(r) >= 10) == 4


def test_c49_s1_coverage_rises_with_depth_from_19_to_50_percent_and_nothing_is_proven_impossible():
    def coverage(ply):
        live = [r for r in _rows() if r["ply"] == ply and not r["trivial"]]
        return len(_found(ply)) / len(live), len(live)

    assert coverage(12) == (11 / 58, 58) and coverage(14) == (25 / 50, 50)
    assert not any(r["status"] == "impossible" for r in _rows())


def test_c49_s1_the_time_budget_was_not_hard_eight_searches_overran_it_up_to_tenfold():
    cap = SPEC["budget"]["seconds"]
    over = [r for r in _rows() if r["status"] == "budget" and r["seconds"] > cap + 60]
    assert len(over) == 8 and max(r["seconds"] for r in over) > 3000
    slow_found = [r for r in _found(14) if r["seconds"] > cap]
    assert len(slow_found) == 3
    live = sum(1 for r in _rows() if r["ply"] == 14 and not r["trivial"])
    assert (len(_found(14)) - len(slow_found)) / live >= 0.44
