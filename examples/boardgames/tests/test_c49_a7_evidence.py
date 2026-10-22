"""§3.6 A7 — PRE-REGISTERED claims on F2's 16 heaviest positions built with L5's rule and pipeline keeping leaves already at
10x (not 30x), judged by harness.floor_a7 from evidence/c49_a7.json.gz (scripts/c4_strategy_s2.py --spec
harness.floor_a7) against B1's control (evidence/c49_b1_control.json.gz): at least 14 of the 16 complete within 2 hours,
and they take no more than nine tenths of the control's 104,860 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_a7 import SPEC, a7_report, a7_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a7.json.gz"
FILES = (FILE, "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def test_c49_a7_14_of_16_complete_within_2_hours_accepting_at_10x():
    assert a7_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_a7_completion_is_undecidable():
    assert a7_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_a7_accepting_at_10x_makes_the_heavy_strategies_a_tenth_smaller():
    assert a7_size(_evidence(), _evidence("c49_b1_control.json.gz"))["verdict"] == "supported"


def test_c49_a7_the_size_is_undecidable():
    assert a7_size(_evidence(), _evidence("c49_b1_control.json.gz"))["verdict"] in {"inconclusive", "not_run"}
