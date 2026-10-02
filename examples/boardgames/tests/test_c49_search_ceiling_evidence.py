"""§C.49 D2, read after the data — more search does NOT make the Connect-4 opening labels right: the share of opening
positions where the net's own search prefers an optimal move does not rise from 200 to 20,000 sims, and at the first
moves even 100,000 sims is right barely half the time. Read from evidence/c49_D2_opening.json.gz with
harness.floor_opening.opening_report (registered after the data, as a description)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_opening import opening_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_D2_opening.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_a_hundred_times_more_search_does_not_fix_the_opening_labels():
    r = opening_report(load_evidence(EVIDENCE / FILE))
    assert r["integrity"] == []
    assert r["pooled"]["20000"] <= r["pooled"]["200"] and max(r["pooled"].values()) < 0.9
    shallow = [r["by_ply"][b][p] for b in ("200", "2000", "20000") for p in ("0", "2")]
    assert max(shallow) <= 0.6 and r["deep_share"] < 0.6
