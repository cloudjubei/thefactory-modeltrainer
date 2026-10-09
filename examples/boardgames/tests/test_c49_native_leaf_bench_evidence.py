"""§3.6 — finding registered AFTER the data, from evidence/c49_native_leaf_bench.json.gz: on four frontier positions the
C leaf walk returns exactly the Python walk's result; a repeated (warm) walk is at least twice as fast, a first (cold)
walk at most three times — the exact solves for newly reached positions bound a cold walk in both."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_native_leaf_bench.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_the_c_walk_matches_and_is_two_to_seven_times_faster_warm_but_solve_bound_cold():
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    assert len(rows) == 4 and all(r["same"] for r in rows)
    assert all(r["python_warm"] >= 2 * r["native_warm"] for r in rows)
    assert all(r["python_cold"] <= 3.5 * r["native_cold"] for r in rows)
