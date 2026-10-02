"""§C.50 E2 — the PRE-REGISTERED claim that a proven-minimal, position-sound tic-tac-toe playbook exists within the
declared rule language, read from evidence/c50_E2_minimal_playbook.json.gz (scripts/minimal_playbook.py). The claim's
proof passes only on SUPPORTED; its undecidable proof passes on INCONCLUSIVE. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c50_E2_minimal_playbook.json.gz"
FILES = (FILE,)
MAX_RULES = 12
POSITIONS = 4520
PREDICATES = ["wins", "blocks", "gives_win", "makes_threat", "forks", "opp_fork_at", "gives_fork", "centre",
              "corner", "side", "opposite_corner"]
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _verdict() -> str:
    e = load_evidence(EVIDENCE / FILE)
    assert e["config"] == {"predicates": PREDICATES, "max_rules": MAX_RULES, "positions": POSITIONS}
    r, check = e["result"], e["verification"]
    if r["k"] is not None:
        perfect = check["positions"] == POSITIONS and check["covered"] == check["correct"] == POSITIONS
        minimal = r["proven_minimal"] and r["literals_proven_minimal"] and r["impossible_up_to"] == r["k"] - 1
        return "supported" if perfect and minimal and r["k"] <= MAX_RULES else "inconclusive"
    return "refuted" if r["inseparable"] or r["impossible_up_to"] >= MAX_RULES else "inconclusive"


def test_c50_E2_a_proven_minimal_position_sound_playbook_of_at_most_12_rules_plays_every_position_optimally():
    assert _verdict() == "supported"


def test_c50_E2_is_undecidable():
    assert _verdict() == "inconclusive"
