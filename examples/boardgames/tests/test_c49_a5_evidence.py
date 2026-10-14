"""§3.6 A5 — PRE-REGISTERED claim on A4's eight frontier positions built, improved and improved with lookahead as A4,
then their costliest leaves with exceptions re-searched for 120 s each in the time left and improved with lookahead
once more, judged by harness.floor_a5 from evidence/c49_a5.json.gz (scripts/c4_strategy_s2.py --spec
harness.floor_a5): that cuts the total the lookahead pass left by at least 10%."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_a5 import SPEC, a5_further

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a5.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_a5_research_cuts_what_lookahead_left_by_10_percent():
    assert a5_further(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_a5_the_research_cut_is_undecidable():
    assert a5_further(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
