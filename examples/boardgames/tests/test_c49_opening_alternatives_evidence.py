"""§3.6 — finding registered AFTER the data, from evidence/c49_opening_alternatives.json.gz
(scripts/c4_opening_alternatives.py): at ply 6 the opening has little room to steer around its heavy frontier positions.
Of the 13 decisions that lead to one of the 12 sampled positions of 5,000 bits or more, 6 have a single winning move,
and at the other 7 every alternative leaves 7 replies the empty map does not win — never fewer than the move chosen."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_opening_alternatives.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_the_decisions_above_the_heavy_positions_have_no_lighter_alternative():
    e = load_evidence(EVIDENCE / FILE)
    heavy = {int(i) for i, n in e["built"].items() if n >= 5000}
    assert len(e["decisions"]) == 148 and len(heavy) == 12
    above = [d for d in e["decisions"] if heavy & set(d["unwon"][str(d["move"])])]
    assert len(above) == 13
    assert sum(len(d["winning"]) == 1 for d in above) == 6
    for d in (d for d in above if len(d["winning"]) > 1):
        chosen = len(d["unwon"][str(d["move"])])
        others = [len(v) for a, v in d["unwon"].items() if int(a) != d["move"]]
        assert all(n == 7 for n in others) and min(others) >= chosen
