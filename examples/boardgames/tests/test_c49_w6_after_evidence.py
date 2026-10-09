"""§3.6 W6 — finding registered AFTER the data, from evidence/c49_w6.json.gz and evidence/c49_w5b.json.gz (the same
build settings on W1's first 40 seeded frontier positions): the frontier splits in two. Half the 32 (16) take under
200 bits — together under 1% of the sample's bits — and the rest carry the size, 84% of all bits being exceptions;
W5b's 8 split the same way (4 under 200 bits), and the 40 together project the whole game to 3.93M bits."""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w6.json.gz"
FILES = (FILE, "c49_w5b.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_w6_half_the_frontier_is_tiny_and_exceptions_are_most_of_the_rest():
    w6 = load_evidence(EVIDENCE / FILE)
    w5b = load_evidence(EVIDENCE / "c49_w5b.json.gz")
    assert all(r["complete"] for e in (w6, w5b) for r in e["roots"])
    sizes = [r["bits"]["nodes"] for r in w6["roots"]]
    tiny = [s for s in sizes if s < 200]
    assert len(tiny) == 16 and sum(tiny) / sum(sizes) < 0.01
    assert sum(r["bits"]["exception_bits"] for r in w6["roots"]) / sum(sizes) > 0.83
    assert sum(r["bits"]["nodes"] < 200 for r in w5b["roots"]) == 4
    opening = w6["opening"]
    both = sizes + [r["bits"]["nodes"] for r in w5b["roots"]]
    projected = opening["moves"] * 4 + opening["trivial"] + opening["frontier"] * statistics.mean(both)
    assert round(projected) == 3_928_807
