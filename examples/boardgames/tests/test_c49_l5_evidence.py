"""§3.6 L5 — PRE-REGISTERED claims on L4's 32 positions (F1's 16 heaviest, 16 spread over the rest) built with L2's rule
and pipeline reading the safe moves on unlevelled cells as a last level, judged by harness.floor_l5 from
evidence/c49_l5.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_l5) against L4's control
(evidence/c49_l4_control.json.gz): at least 28 of the 32 complete within 2 hours, and they take no more than four
fifths of the control's 130,061 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l5 import SPEC, l5_report, l5_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l5.json.gz"
FILES = (FILE, "c49_l4_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return l5_size(_evidence(), _evidence("c49_l4_control.json.gz"), SPEC)


def test_c49_l5_28_of_32_complete_within_2_hours_reading_unlevelled_cells_last():
    assert l5_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_l5_completion_is_undecidable():
    assert l5_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_l5_reading_unlevelled_cells_last_makes_the_strategies_a_fifth_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_l5_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
