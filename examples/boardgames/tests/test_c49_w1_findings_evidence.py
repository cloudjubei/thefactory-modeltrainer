"""§3.6 W1 — finding registered AFTER the data, from evidence/c49_w1.json.gz (h191 refuted): every completed frontier
strategy is 30-50x under its own table, yet the three largest of the eight sampled strategies carry over 80% of the
sample's bits — the whole game's size is set by what the opening leaves (a few big subtrees), not by how each subtree
is compressed."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_w1_subtrees_compress_alike_but_three_big_ones_carry_most_of_the_bits():
    roots = load_evidence(EVIDENCE / FILE)["roots"]
    done = [r for r in roots if r["complete"]]
    assert len(roots) == 8 and len(done) == 7
    assert all(30 <= 3 * r["own_positions"] / r["bits"]["nodes"] <= 50 for r in done)
    sizes = sorted((r["bits"]["nodes"] for r in roots), reverse=True)
    assert sum(sizes[:3]) > 0.8 * sum(sizes)
