"""§3.6 W2 — finding registered AFTER the data, from evidence/c49_w2.json.gz beside evidence/c49_w1.json.gz (h202,
h203 inconclusive): every one of the 8 sampled frontier strategies is smaller under W2 than under W1 — finished or
not, at most two-thirds of W1's size and on the biggest W1 strategies under a tenth — so even as a lower bound the
projection falls more than tenfold; the library is barely used, so the gain is the root searches'."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_w2.json.gz", "c49_w1.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_w2_every_frontier_strategy_is_smaller_and_the_biggest_by_tenfold():
    w2 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[0])["roots"]}
    w1 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[1])["roots"]}
    assert set(w2) == set(w1) and len(w2) == 8
    assert all(w2[i]["bits"]["nodes"] <= 2 / 3 * w1[i]["bits"]["nodes"] for i in (362, 346, 263, 393, 323, 316, 230))
    assert w2[87]["bits"]["nodes"] < w1[87]["bits"]["nodes"]
    big = sorted(w1, key=lambda i: -w1[i]["bits"]["nodes"])[:3]
    assert all(w2[i]["bits"]["nodes"] < w1[i]["bits"]["nodes"] / 10 for i in big)
    assert sum(r["bits"]["nodes"] for r in w2.values()) * 10 < sum(r["bits"]["nodes"] for r in w1.values())
    later = [r for r in w2.values() if r["shared"]]
    assert sum(r["library_leaves"] for r in later) <= 0.05 * sum(r["bits"]["leaves"] for r in later)


def test_c49_w2_every_frontier_strategy_is_smaller_seven_by_at_least_30_percent_the_biggest_tenfold():
    w2 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[0])["roots"]}
    w1 = {r["index"]: r for r in load_evidence(EVIDENCE / FILES[1])["roots"]}
    assert set(w2) == set(w1) and len(w2) == 8
    assert all(w2[i]["bits"]["nodes"] < w1[i]["bits"]["nodes"] for i in w2)
    assert sum(1 for i in w2 if w2[i]["bits"]["nodes"] <= 0.7 * w1[i]["bits"]["nodes"]) == 7
    big = sorted(w1, key=lambda i: -w1[i]["bits"]["nodes"])[:3]
    assert all(w2[i]["bits"]["nodes"] < w1[i]["bits"]["nodes"] / 10 for i in big)
    assert sum(r["bits"]["nodes"] for r in w2.values()) * 10 < sum(r["bits"]["nodes"] for r in w1.values())
    later = [r for r in w2.values() if r["shared"]]
    assert sum(r["library_leaves"] for r in later) <= 0.05 * sum(r["bits"]["leaves"] for r in later)
