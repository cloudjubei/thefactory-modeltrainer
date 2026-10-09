"""§3.6 P2 — finding registered AFTER the data, from evidence/c49_p2.json.gz beside W1 and P1 (h197, h198
inconclusive): with one long search at the root, #323 completes at a tenth of W1's bits and under P1's, but #316 and
#230 make at most 16 searches each in 3 hours — fewer than P1's and a fraction of W1's — so the build near the root,
not the leaf search's budget, is what keeps the big frontier positions from finishing."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_p2.json.gz", "c49_p1.json.gz", "c49_w1.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _roots(name):
    return {r["index"]: r for r in load_evidence(EVIDENCE / name)["roots"]}


def test_c49_p2_a_root_search_shrinks_323_ten_fold_but_the_other_two_barely_search():
    p2, p1, w1 = _roots(FILES[0]), _roots(FILES[1]), _roots(FILES[2])
    a = p2[323]
    assert a["complete"] and a["checked"] and a["moves_win"] and a["leaves_certified"]
    assert a["bits"]["nodes"] <= w1[323]["bits"]["nodes"] / 10 and a["bits"]["nodes"] < p1[323]["bits"]["nodes"]
    for i in (316, 230):
        assert not p2[i]["complete"] and p2[i]["searches"]["tried"] <= 16
        assert p2[i]["searches"]["tried"] < p1[i]["searches"]["tried"] < w1[i]["searches"]["tried"]
