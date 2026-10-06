"""§3.6 option 1 — PRE-REGISTERED claim: local search over priority maps (harness.steady_local, 8 levels, 30 min each)
finds verified steady states for at least 3 of the 8 ply-10 positions the S1 pilot left out of budget, from
evidence/c49_s1_local_probe.json.gz (scripts/c4_steady_local_probe.py); SAT under the same conditions found 1 (h162).
Found maps must pass the solver-free walk and certify."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_local_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _found() -> int:
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    assert len(rows) == 8 and all(r["verified"] and r["certified"] for r in rows if r["status"] == "found")
    return sum(1 for r in rows if r["status"] == "found")


def test_c49_s1_local_search_finds_steady_states_for_3_of_8_ply_10_positions():
    assert _found() >= 3


def test_c49_s1_the_local_search_reading_is_undecidable():
    assert _found() == 2
