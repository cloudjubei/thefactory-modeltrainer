"""§3.6 A5 — finding registered AFTER the data, from evidence/c49_a5.json.gz: after improving with lookahead little is
left for longer leaf searches — the eight positions held only 29 leaves with exceptions to re-search between them (3
kept) — because exceptions are now 14% of their bits and maps 64%: the maps are the cost to attack next."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a5.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_a5_little_is_left_to_research_because_maps_are_now_most_of_the_bits():
    roots = load_evidence(EVIDENCE / FILE)["roots"]
    assert len(roots) == 8 and all(r["complete"] for r in roots)
    assert sum(r["improved"]["research"]["tried"] for r in roots) == 29
    assert sum(r["improved"]["research"]["kept"] for r in roots) == 3
    total = sum(r["bits"]["nodes"] for r in roots)
    assert 0.13 < sum(r["bits"]["exception_bits"] for r in roots) / total < 0.15
    assert 0.63 < sum(r["bits"]["map_bits"] for r in roots) / total < 0.65
