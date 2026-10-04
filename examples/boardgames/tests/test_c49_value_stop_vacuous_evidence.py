"""§C.49 T11b — DESCRIPTIVE, registered after the data: on tic-tac-toe the value-aware reading never differed from
the share reading, so T11's SUPPORTED verdict (h101) says nothing about the value rule itself. Read from
evidence/c49_T11_tree_value_stop_weak.json.gz. The register pins this file."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T11_tree_value_stop_weak.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_T11b_the_value_reading_equals_the_share_reading_on_every_walked_iteration_of_every_seed():
    seeds = load_evidence(EVIDENCE / FILE)["seeds"]
    walked = [h for r in seeds for h in r["history"][1:-1]]
    assert len(seeds) == 10 and len(walked) == 290 and sum(h["tree_disagreements"] for h in walked) == 569
    assert all(h["tree_disagreements"] == h["tree_disagreements_share"] for h in walked)
