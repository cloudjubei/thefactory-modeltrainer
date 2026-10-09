"""§3.6 W4 — PRE-REGISTERED claims on W3 rebuilt with the leaves walked in C, judged by harness.floor_w4.w4_report from
evidence/c49_w4.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_w4) against evidence/c49_w3.json.gz: #316
finishes in at most half W3's time; and #230, unfinished in W3's 6 hours, finishes."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w4 import SPEC, w4_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w4.json.gz"
FILES = (FILE, "c49_w3.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return w4_report(load_evidence(EVIDENCE / FILE), SPEC, load_evidence(EVIDENCE / FILES[1]))


def test_c49_w4_the_c_walk_halves_the_time_to_finish_316():
    assert _report()["time"]["verdict"] == "supported"


def test_c49_w4_the_time_change_is_undecidable():
    assert _report()["time"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w4_the_c_walk_finishes_230():
    assert _report()["unfinished"]["verdict"] == "supported"
