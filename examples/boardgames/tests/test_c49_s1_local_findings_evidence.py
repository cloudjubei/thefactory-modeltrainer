"""§3.6 option 1 — finding registered AFTER the data, from evidence/c49_s1_local_probe.json.gz (h166 refuted): local
search ends one failing line short on seven of the eight ply-10 positions and two short on the eighth."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_local_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_s1_local_search_ends_one_line_short_on_7_of_8_positions():
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    assert all(r["status"] == "budget" for r in rows)
    assert Counter(r["best_failures"] for r in rows) == {1: 7, 2: 1}
