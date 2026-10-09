"""§3.6 A1 — finding registered AFTER the data, from evidence/c49_a1.json.gz against evidence/c49_w6.json.gz: a fixed
threshold for keeping exceptions cuts both ways. Splitting more shrinks three of the four heavy positions (#594 by 81%)
but makes #565 4.7x larger — its one root map with exceptions covered 43,562 positions, the split strategy reaches
321,970. Taking the smaller of the two builds per position would give 28,248 bits, under 39% of W6's 72,883."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a1.json.gz"
FILES = (FILE, "c49_w6.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_a1_a_fixed_threshold_cuts_both_ways_and_the_smaller_build_per_position_is_under_two_fifths():
    a1 = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    w6 = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_w6.json.gz")["roots"]}
    assert all(r["complete"] for r in a1.values()) and set(a1) == {95, 565, 594, 535}
    smaller = [i for i in a1 if a1[i]["bits"]["nodes"] < w6[i]["bits"]["nodes"]]
    assert sorted(smaller) == [95, 535, 594]
    assert a1[594]["bits"]["nodes"] < 0.2 * w6[594]["bits"]["nodes"]
    assert a1[565]["bits"]["nodes"] > 4.6 * w6[565]["bits"]["nodes"]
    assert a1[565]["own_positions"] > 7 * w6[565]["own_positions"]
    best = sum(min(a1[i]["bits"]["nodes"], w6[i]["bits"]["nodes"]) for i in a1)
    assert best == 28_248 and best / sum(w6[i]["bits"]["nodes"] for i in a1) < 0.39
