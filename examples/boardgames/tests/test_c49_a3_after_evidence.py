"""§3.6 A3 — finding registered AFTER the data, from evidence/c49_a3.json.gz: improving anytime is cheap and cuts deep.
Each of the four heavy positions' improvement queue ran dry within 15 minutes of its threshold build completing, the
longest build taking 45 minutes, and improving cut each completed build by 55-82% — 62,621 bits to 16,740 in all."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a3.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_a3_improving_runs_dry_within_15_minutes_and_cuts_each_build_by_more_than_half():
    roots = load_evidence(EVIDENCE / FILE)["roots"]
    assert len(roots) == 4 and all(r["complete"] for r in roots)
    for r in roots:
        imp = r["improved"]
        assert r["build_seconds"] - imp["complete_seconds"] < 900 and r["build_seconds"] < 2700
        assert 0.18 <= r["bits"]["nodes"] / imp["complete_bits"] <= 0.45
    assert sum(r["improved"]["complete_bits"] for r in roots) == 62_621
    assert sum(r["bits"]["nodes"] for r in roots) == 16_740


def test_c49_a3_improving_runs_dry_within_15_minutes_and_cuts_each_build_by_54_to_82_percent():
    roots = load_evidence(EVIDENCE / FILE)["roots"]
    assert len(roots) == 4 and all(r["complete"] for r in roots)
    for r in roots:
        imp = r["improved"]
        assert r["build_seconds"] - imp["complete_seconds"] < 900 and r["build_seconds"] < 2700
        assert 0.17 <= r["bits"]["nodes"] / imp["complete_bits"] <= 0.46
    assert sum(r["improved"]["complete_bits"] for r in roots) == 62_621
    assert sum(r["bits"]["nodes"] for r in roots) == 16_740
