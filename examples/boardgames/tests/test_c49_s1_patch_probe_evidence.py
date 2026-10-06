"""§3.6 — PRE-REGISTERED claims on map-plus-exceptions leaves at ply 10, from evidence/c49_s1_patch_probe.json.gz
(scripts/c4_steady_patch_probe.py): local search (30 min) then at most 100 exceptions give a verified leaf for at least
6 of the 8 ply-10 positions the S1 pilot left out of budget; and the median leaf is at least 10x smaller than its
own positions as a 3-bit table. Leaves must pass the solver-free walk with their exceptions and certify."""
from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_patch_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _leaves() -> list:
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    assert len(rows) == 8
    done = [r for r in rows if r["patch"] == "patched"]
    assert all(r["verified"] and r["certified"] for r in done)
    return done


def _median_compression():
    return statistics.median(3 * r["own_positions"] / r["bits"] for r in _leaves())


def test_c49_s1_map_plus_exceptions_leaves_for_6_of_8_ply_10_positions():
    assert len(_leaves()) >= 6


def test_c49_s1_the_leaf_count_is_undecidable():
    assert 3 <= len(_leaves()) <= 5


def test_c49_s1_map_plus_exceptions_leaves_are_10_times_smaller_than_their_table():
    assert _leaves() and _median_compression() >= 10


def test_c49_s1_the_leaf_compression_is_undecidable():
    assert _leaves() and 1 <= _median_compression() < 10
