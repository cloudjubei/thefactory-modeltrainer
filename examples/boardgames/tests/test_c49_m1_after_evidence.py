"""§3.6 M1 — finding registered AFTER the data, from evidence/c49_m1.json.gz: F1's maps are near their information
content under simple exact codes. One code for every map — the levelled cells chosen among the empty ones, each level
in full — takes 282,825 bits against the builder's 292,054 (3.2% less); the cheapest code per map plus a 2-bit selector
takes more, 284,818, the selector costing more than switching codes gains; levels by count take 288,075 and by rank
291,758 (most maps use seven or eight levels). The map bits are a matter of how many maps and how many levelled cells,
not of how they are written."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_m1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_m1_one_simple_code_beats_choosing_per_map_and_saves_three_percent():
    totals = load_evidence(EVIDENCE / FILE)["totals"]
    assert totals == {"bitmap": 292_054, "chosen": 282_825, "ranks": 291_758, "counts": 288_075, "best": 284_818}
    assert totals["chosen"] < totals["best"] and 1 - totals["chosen"] / totals["bitmap"] < 0.033
