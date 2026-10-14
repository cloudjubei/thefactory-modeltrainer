"""§3.6 L2 — PRE-REGISTERED claims on W6's 32 frontier positions built with W8's pipeline and 8 levels, the rule's
levels read with ties to the lowest move (the first level holding any safe move decides), judged by harness.floor_l2
from evidence/c49_l2.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_l2): at least 28 of the 32 complete
within 2 hours, and the 32 take no more than four fifths of the smaller run of the exact rule (W7, 69,847 bits)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l2 import SPEC, l2_report, l2_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l2.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_l2_28_of_32_complete_within_2_hours():
    assert l2_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_l2_completion_is_undecidable():
    assert l2_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_l2_ties_to_the_lowest_make_the_strategies_a_fifth_smaller():
    assert l2_size(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_l2_the_size_is_undecidable():
    assert l2_size(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
