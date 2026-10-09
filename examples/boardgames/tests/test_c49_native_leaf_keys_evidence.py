"""§3.6 — PRE-REGISTERED claim on building the C walk's exception keys only when read and reusing its output buffers,
from evidence/c49_native_leaf_bench_keys.json.gz beside h210's benchmark (evidence/c49_native_leaf_bench_lazy.json.gz)
on the same 4 frontier positions: results still equal the Python walk's, and the median warm walk is at least 1.5x
faster than before."""
from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_native_leaf_bench_keys.json.gz"
FILES = (FILE, "c49_native_leaf_bench_lazy.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _speedup() -> float:
    new = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["rows"]}
    old = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[1])["rows"]}
    assert set(new) == set(old) and all(r["same"] for r in new.values())
    return statistics.median(old[i]["native_warm"] / new[i]["native_warm"] for i in new)


def test_c49_lazy_keys_make_warm_walks_half_again_faster():
    assert _speedup() >= 1.5


def test_c49_the_warm_speedup_is_undecidable():
    assert 1 <= _speedup() < 1.5
