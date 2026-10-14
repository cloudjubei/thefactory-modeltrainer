"""§3.6 A4 — finding registered AFTER the data, from evidence/c49_a4.json.gz: building, improving and improving with
lookahead cut the eight positions' completed builds by 63% (82,138 bits to 29,989), and every build was done within
72 minutes — at least 48 minutes of each 2-hour cap left unused."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a4.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_a4_the_pipeline_cuts_completed_builds_by_63_percent_inside_72_minutes():
    roots = load_evidence(EVIDENCE / FILE)["roots"]
    assert len(roots) == 8 and all(r["complete"] for r in roots)
    completed = sum(r["improved"]["complete_bits"] for r in roots)
    final = sum(r["bits"]["nodes"] for r in roots)
    assert (completed, final) == (82_138, 29_989) and 0.63 <= 1 - final / completed < 0.64
    assert max(r["build_seconds"] for r in roots) < 72 * 60
