"""§3.6 S1 pilots — findings registered AFTER the data, from evidence/c49_s1_pilot.json.gz (plies 10 and 12) and
evidence/c49_s1_pilot14.json.gz (plies 12 and 14), each ten positions per ply outside the registered sample: with the
registered language and budget our own search found no steady state at plies 10-12, and one at ply 14."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_s1_pilot.json.gz", "c49_s1_pilot14.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _statuses() -> dict:
    out: dict = {}
    for f in FILES:
        for r in load_evidence(EVIDENCE / f)["results"]:
            out.setdefault(r["ply"], {})[(f, r["index"])] = r
    return {ply: Counter(r["status"] for r in rows.values()) for ply, rows in out.items()}


def test_c49_s1_pilots_find_nothing_at_plies_10_and_12_and_one_state_at_ply_14():
    s = _statuses()
    assert s[10] == {"trivial": 2, "budget": 8}
    assert s[12] == {"trivial": 14, "impossible": 2, "budget": 4}
    assert s[14] == {"trivial": 5, "found": 1, "budget": 4}
    found = [r for f in FILES for r in load_evidence(EVIDENCE / f)["results"] if r["status"] == "found"]
    assert all(r["verified"] and r["certified"] for r in found)
