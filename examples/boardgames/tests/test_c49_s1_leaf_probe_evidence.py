"""§3.6 — PRE-REGISTERED claims on map-plus-exceptions leaves searched by their size at ply 10, from
evidence/c49_s1_leaf_probe.json.gz (scripts/c4_steady_leaf_probe.py): on the 8 ply-10 positions the S1 pilot left out
of budget, local search scored by leaf size (30 min, 8 levels) gives a median leaf at least 10x smaller than its own
positions as a 3-bit table, and at least 2x smaller than the empty map's leaf it starts from. Every leaf must pass
the solver-free walk with its exceptions and certify."""
from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_leaf_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _rows() -> list:
    data = load_evidence(EVIDENCE / FILE)
    assert data["settings"]["only"] is None and data["settings"]["seconds"] == 1800.0
    rows = data["rows"]
    assert len(rows) == 8 and all(r["bits"] is not None and r["verified"] and r["certified"] for r in rows)
    return rows


def _compression():
    return statistics.median(3 * r["own_positions"] / r["bits"] for r in _rows())


def _gain():
    return statistics.median(r["start_bits"] / r["bits"] for r in _rows())


def test_c49_s1_size_searched_leaves_are_10_times_smaller_than_their_table():
    assert _compression() >= 10


def test_c49_s1_the_size_searched_leaf_compression_is_undecidable():
    assert 3 <= _compression() < 10


def test_c49_s1_the_search_halves_the_empty_map_s_leaf():
    assert _gain() >= 2


def test_c49_s1_the_search_s_gain_is_undecidable():
    assert 1.2 <= _gain() < 2
