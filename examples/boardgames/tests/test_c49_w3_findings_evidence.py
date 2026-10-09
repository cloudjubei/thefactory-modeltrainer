"""§3.6 W3 — finding registered AFTER the data, from evidence/c49_w3.json.gz beside evidence/c49_w2.json.gz (h207
refuted): given more time, W2's two unfinished strategies grew far past their 2-hour partial sizes — #316 finished
nearly tenfold larger, #230 is more than eightfold larger and still unfinished — so a partial size undercounts the
finished strategy badly, and W2's 533K lower bound was far below the whole game's size."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_w3.json.gz", "c49_w2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_w3_two_hour_partials_undercount_the_finished_strategies_tenfold():
    w3 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[0])["roots"]}
    w2 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[1])["roots"]}
    assert not w2[316]["complete"] and not w2[230]["complete"]
    assert w3[316]["complete"] and w3[316]["checked"] and w3[316]["moves_win"] and w3[316]["leaves_certified"]
    assert w3[316]["bits"]["nodes"] > 9 * w2[316]["bits"]["nodes"]
    assert not w3[230]["complete"] and w3[230]["bits"]["nodes"] > 8 * w2[230]["bits"]["nodes"]
