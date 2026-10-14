"""§3.6 L1 — PRE-REGISTERED claims on W6's 32 frontier positions built with the full pipeline as W8 but with maps of 4
levels at 2 bits a levelled cell, judged by harness.floor_l1 from evidence/c49_l1.json.gz (scripts/c4_strategy_s2.py
--spec harness.floor_l1): at least 28 of the 32 complete within 2 hours, and the 32 take no more than four fifths of
the smaller 8-level run's 69,847 bits (W7; W8: 74,714)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l1 import SPEC, l1_report, l1_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_l1_28_of_32_complete_within_2_hours():
    assert l1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_l1_completion_is_undecidable():
    assert l1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_l1_four_levels_make_the_strategies_a_fifth_smaller():
    assert l1_size(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_l1_the_size_is_undecidable():
    assert l1_size(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
