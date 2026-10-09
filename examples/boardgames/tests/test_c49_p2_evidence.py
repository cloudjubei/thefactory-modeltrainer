"""§3.6 P2 — PRE-REGISTERED claims on rebuilding W1's three largest frontier strategies with one long search for a pure
steady state at each root (30 s below), judged by harness.floor_p2.p2_report (= floor_p1's report) from
evidence/c49_p2.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_p2) against evidence/c49_w1.json.gz: the two
strategies W1 finished come out at no more than half their W1 size together; and all three complete within 3 hours."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_p2 import SPEC, p2_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_p2.json.gz"
FILES = (FILE, "c49_w1.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return p2_report(load_evidence(EVIDENCE / FILE), SPEC, load_evidence(EVIDENCE / FILES[1]))


def test_c49_p2_long_root_searches_halve_w1_s_largest_finished_strategies():
    assert _report()["size"]["verdict"] == "supported"


def test_c49_p2_the_size_change_is_undecidable():
    assert _report()["size"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_p2_all_three_complete_within_3_hours():
    assert _report()["complete"]["verdict"] == "supported"


def test_c49_p2_completion_is_undecidable():
    assert _report()["complete"]["verdict"] in {"inconclusive", "not_run"}
