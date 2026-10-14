"""§3.6 W8 — finding registered AFTER the data, from evidence/c49_w8.json.gz and evidence/c49_w7.json.gz: between runs
of the same settings the builds swing more than lookahead gains inside them. W8's builds completed at 186,359 bits
against W7's 154,777 (+20%), so the whole game projects higher than W7's though lookahead cut 16% of what plain
improvement left; keeping the smaller of W7's and W8's certified strategy per position gives 60,235 bits — the whole
game at 1.26M, 14% under W7's projection."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w8.json.gz"
FILES = (FILE, "c49_w7.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_w8_runs_swing_more_than_lookahead_gains_and_the_better_of_two_runs_projects_1_26m():
    w8 = load_evidence(EVIDENCE / FILE)
    w7 = load_evidence(EVIDENCE / "c49_w7.json.gz")
    assert all(r["complete"] for e in (w7, w8) for r in e["roots"])
    completed = {name: sum(r["improved"]["complete_bits"] for r in e["roots"]) for name, e in (("w7", w7), ("w8", w8))}
    assert completed == {"w7": 154_777, "w8": 186_359}
    plain = sum(r["improved"]["plain_bits"] for r in w8["roots"])
    final = sum(r["bits"]["nodes"] for r in w8["roots"])
    assert (plain, final) == (88_773, 74_714)
    w7_bits = {r["index"]: r["bits"]["nodes"] for r in w7["roots"]}
    best = sum(min(r["bits"]["nodes"], w7_bits[r["index"]]) for r in w8["roots"])
    opening = w8["opening"]
    assert best == 60_235
    assert round(opening["moves"] * 4 + opening["trivial"] + opening["frontier"] * best / 32) == 1_264_128
