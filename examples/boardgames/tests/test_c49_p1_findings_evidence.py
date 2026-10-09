"""§3.6 P1 — finding registered AFTER the data, from evidence/c49_p1.json.gz beside evidence/c49_w1.json.gz (h194,
h195 inconclusive): where long searches near the root succeed the strategy shrinks several-fold — #323 completes at
under a fifth of W1's bits, playing under a fifth of W1's positions — but on the other two the long searches use the
time: at most a quarter of W1's searches are made and neither finishes in 3 hours."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_p1.json.gz", "c49_w1.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_p1_a_root_map_shrinks_323_five_fold_but_long_searches_starve_the_other_two():
    p1 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[0])["roots"]}
    w1 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[1])["roots"]}
    a, b = p1[323], w1[323]
    assert a["complete"] and a["checked"] and a["moves_win"] and a["leaves_certified"] and b["complete"]
    assert a["bits"]["nodes"] < b["bits"]["nodes"] / 5 and a["own_positions"] < b["own_positions"] / 5
    for i in (316, 230):
        assert not p1[i]["complete"] and p1[i]["searches"]["tried"] <= w1[i]["searches"]["tried"] / 4
