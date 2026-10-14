"""§3.6 L3 — PRE-REGISTERED claims on W6's 32 frontier positions built exactly as L2 but with two-ply safety (a move
is not safe when the opponent can answer it with a forced win in two), judged by harness.floor_l3 from
evidence/c49_l3.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_l3): at least 28 of the 32 complete within 2
hours, and the 32 take no more than four fifths of L2's 14,666 bits on the same positions."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l3 import SPEC, l3_report, l3_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l3.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_l3_28_of_32_complete_within_2_hours():
    assert l3_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_l3_completion_is_undecidable():
    assert l3_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_l3_two_ply_safety_makes_the_strategies_a_fifth_smaller():
    assert l3_size(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_l3_the_size_is_undecidable():
    assert l3_size(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
