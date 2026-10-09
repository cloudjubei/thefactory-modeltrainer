"""§3.6 W4 — finding registered AFTER the data, from evidence/c49_w4.json.gz beside evidence/c49_w3.json.gz (h211,
h212 refuted): walking the leaves in C does not shorten a time-bounded build — #316 takes within 5% of W3's time — and
both builds make fewer searches than W3's, so the time a faster walk frees inside the searches (which run to their
budgets) does not reach the build; the time goes elsewhere."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_w4.json.gz", "c49_w3.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_w4_the_c_walk_leaves_build_time_unchanged_and_searches_fewer():
    w4 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[0])["roots"]}
    w3 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[1])["roots"]}
    assert w4[316]["complete"] and w3[316]["complete"]
    assert 0.95 <= w4[316]["build_seconds"] / w3[316]["build_seconds"] <= 1.05
    assert all(w4[i]["searches"]["tried"] < w3[i]["searches"]["tried"] for i in (316, 230))
