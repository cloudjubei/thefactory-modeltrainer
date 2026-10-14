"""§3.6 L1 — dead end registered AFTER the data, from evidence/c49_l1.json.gz against evidence/c49_w8.json.gz: fewer,
cheaper levels lose. With 4 levels at 2 bits the 32 strategies need 2,132 maps against W8's 586 with 8 levels, and are
larger on 25 of the 32 positions; 9 did not finish in 2 hours — a map's expressiveness, not its cell cost, binds."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l1.json.gz"
FILES = (FILE, "c49_w8.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_l1_four_levels_need_more_than_three_times_the_maps_and_are_larger_on_most_positions():
    l1 = load_evidence(EVIDENCE / FILE)["roots"]
    w8 = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_w8.json.gz")["roots"]}
    assert sum(r["bits"]["maps"] for r in l1) == 2_132 and sum(r["bits"]["maps"] for r in w8.values()) == 586
    assert sum(r["bits"]["nodes"] > w8[r["index"]]["bits"]["nodes"] for r in l1) == 25
    assert sum(not r["complete"] for r in l1) == 9
