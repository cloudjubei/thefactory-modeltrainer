"""§3.6 W7 — finding registered AFTER the data, from evidence/c49_w7.json.gz against evidence/c49_w6.json.gz: W7's cut
from W6 has two parts. With W6's settings, W7's builds complete at 154,777 bits against W6's 205,322 — run-to-run
variance, per position from a ninth to two and a half times W6's — and improving then cuts them to 69,847, 45% of
what completed: improving alone takes the projection from 3.25M bits to 1.47M. Four positions kept no split: their
one candidate, a leaf at the root, split larger one level down, so nothing below it was tried."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w7.json.gz"
FILES = (FILE, "c49_w6.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_w7_the_cut_is_build_variance_and_improvement_and_four_roots_got_stuck_at_one_split():
    w7 = load_evidence(EVIDENCE / FILE)
    w6 = {r["index"]: r["bits"]["nodes"] for r in load_evidence(EVIDENCE / "c49_w6.json.gz")["roots"]}
    roots = w7["roots"]
    completed = sum(r["improved"]["complete_bits"] for r in roots)
    final = sum(r["bits"]["nodes"] for r in roots)
    assert (sum(w6.values()), completed, final) == (205_322, 154_777, 69_847)
    ratios = [r["improved"]["complete_bits"] / w6[r["index"]] for r in roots if w6[r["index"]] > 200]
    assert min(ratios) < 0.1 and max(ratios) > 2.4
    opening = w7["opening"]
    base = opening["moves"] * 4 + opening["trivial"]
    assert round((base + 671 * completed / 32) / 1e4) == 325 and round((base + 671 * final / 32) / 1e4) == 147
    stuck = sorted(r["index"] for r in roots
                   if r["bits"]["nodes"] > 200 and r["improved"]["tried"] == 1 and r["improved"]["kept"] == 0)
    assert stuck == [95, 271, 374, 512]
    assert all(r["bits"]["moves"] == 0 and r["bits"]["leaves"] == 1 for r in roots if r["index"] in stuck)
