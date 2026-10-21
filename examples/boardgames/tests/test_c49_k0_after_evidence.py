"""§3.6 K0 — finding registered AFTER the data, from evidence/c49_k0.json.gz: two-phase maps help the 30-s leaf search
only on leaves far too large to keep. Over the 43 positions where both searches found a leaf, the two-phase leaves total
976,657 bits against 1,832,901 — but the gap is a few leaves of tens to hundreds of thousands of bits, thousands of
exceptions each; on the 26 where the one-phase leaf is under 5,000 bits the two-phase one is larger on 17 and smaller
on 4 (24,160 bits against 15,904)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_k0.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_k0_two_phase_maps_help_only_on_leaves_far_too_large_to_keep():
    rows = [r for r in load_evidence(EVIDENCE / FILE)["rows"] if r["one"]["bits"] and r["two"]["bits"]]
    assert len(rows) == 43
    assert (sum(r["two"]["bits"] for r in rows), sum(r["one"]["bits"] for r in rows)) == (976_657, 1_832_901)
    small = [r for r in rows if r["one"]["bits"] < 5_000]
    assert len(small) == 26
    assert sum(r["two"]["bits"] > r["one"]["bits"] for r in small) == 17
    assert sum(r["two"]["bits"] < r["one"]["bits"] for r in small) == 4
    assert (sum(r["two"]["bits"] for r in small), sum(r["one"]["bits"] for r in small)) == (24_160, 15_904)
