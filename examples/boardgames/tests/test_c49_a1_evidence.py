"""§3.6 A1 — PRE-REGISTERED claims on four of W6's heavy frontier positions (#95, #565, #594, #535) rebuilt as W6 built
them but keeping a leaf with exceptions only at >= 100x under its table (W6: 30x), judged by harness.floor_a1 from
evidence/c49_a1.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_a1): all four complete within 2 hours, and
together they take at most 4/5 of W6's 72,883 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_a1 import SPEC, a1_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_a1_all_four_complete_within_2_hours():
    assert a1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_a1_completion_is_undecidable():
    assert a1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_a1_splitting_more_makes_them_a_fifth_smaller():
    assert a1_report(_evidence(), SPEC)["size"]["verdict"] == "supported"


def test_c49_a1_the_size_change_is_undecidable():
    assert a1_report(_evidence(), SPEC)["size"]["verdict"] in {"inconclusive", "not_run"}
