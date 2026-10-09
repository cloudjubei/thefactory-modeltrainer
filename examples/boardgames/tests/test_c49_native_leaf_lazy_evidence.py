"""§3.6 — PRE-REGISTERED claim on solving only the moves the C leaf walk needs (the map's move where it gives one;
otherwise columns up to the lowest winning one), from evidence/c49_native_leaf_bench_lazy.json.gz beside the earlier
benchmark evidence/c49_native_leaf_bench.json.gz (h208) on the same 4 frontier positions: results still equal the
Python walk's, and the median cold walk is at least twice as fast as before."""
from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_native_leaf_bench_lazy.json.gz"
FILES = (FILE, "c49_native_leaf_bench.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _speedup() -> float:
    new = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["rows"]}
    old = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[1])["rows"]}
    assert set(new) == set(old) and all(r["same"] for r in new.values())
    return statistics.median(old[i]["native_cold"] / new[i]["native_cold"] for i in new)


def test_c49_solving_only_needed_moves_halves_the_cold_walk():
    assert _speedup() >= 2


def test_c49_the_cold_speedup_is_undecidable():
    assert 1 <= _speedup() < 2
