"""§C.50 E2, read after the data — the 11-predicate rule language cannot write a position-sound tic-tac-toe playbook
of up to 60 rules: SAT proves every length 1-60 impossible, though no single position is inseparable. Read from
evidence/c50_E2_beyond_60.json.gz (registered after the data, as a description)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c50_E2_beyond_60.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c50_no_position_sound_playbook_of_up_to_60_rules_exists_in_the_11_predicate_language():
    e = load_evidence(EVIDENCE / FILE)
    assert e["config"]["max_rules"] == 60 and e["config"]["positions"] == 4520 and len(e["config"]["predicates"]) == 11
    r = e["result"]
    assert r["k"] is None and r["impossible_up_to"] == 60 and r["inseparable"] == []
