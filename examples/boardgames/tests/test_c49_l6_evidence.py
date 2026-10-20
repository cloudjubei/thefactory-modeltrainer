"""§3.6 L6 — PRE-REGISTERED claims on L4's 32 positions (F1's 16 heaviest, 16 spread over the rest) built with L5's rule
and pipeline breaking a deciding level's ties by tactics, judged by harness.floor_l6 from evidence/c49_l6.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_l6) against L5 (evidence/c49_l5.json.gz): at least 28 of the 32
complete within 2 hours, and they take no more than nine tenths of L5's 109,017 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l6 import SPEC, l6_report, l6_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l6.json.gz"
FILES = (FILE, "c49_l5.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return l6_size(_evidence(), _evidence("c49_l5.json.gz"), SPEC)


def test_c49_l6_28_of_32_complete_within_2_hours_breaking_ties_by_tactics():
    assert l6_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_l6_completion_is_undecidable():
    assert l6_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_l6_breaking_ties_by_tactics_makes_the_strategies_a_tenth_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_l6_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
