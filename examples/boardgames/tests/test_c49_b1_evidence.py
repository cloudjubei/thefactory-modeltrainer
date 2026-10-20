"""§3.6 B1 — PRE-REGISTERED claims on F2's 16 heaviest positions built with L5's rule and pipeline, the deeper leaf
searches given 120 s instead of 30 s, against a control built the same way with 30 s, judged by harness.floor_b1 from
evidence/c49_b1.json.gz and evidence/c49_b1_control.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_b1, then
--spec harness.floor_b1_control): at least 14 of the 16 complete within 2 hours with the longer searches, and they take
no more than nine tenths of the control's bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_b1 import SPEC, b1_report, b1_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_b1.json.gz"
FILES = (FILE, "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def test_c49_b1_14_of_16_complete_within_2_hours_with_longer_searches():
    assert b1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_b1_completion_is_undecidable():
    assert b1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_b1_longer_searches_make_the_heavy_strategies_a_tenth_smaller():
    assert b1_size(_evidence(), _evidence("c49_b1_control.json.gz"))["verdict"] == "supported"


def test_c49_b1_the_size_is_undecidable():
    assert b1_size(_evidence(), _evidence("c49_b1_control.json.gz"))["verdict"] in {"inconclusive", "not_run"}
