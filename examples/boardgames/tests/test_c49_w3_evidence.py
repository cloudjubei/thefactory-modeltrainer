"""§3.6 W3 — PRE-REGISTERED claims deciding W2's projection, judged by harness.floor_w3.w3_report from
evidence/c49_w3.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_w3) with evidence/c49_w2.json.gz: W2's two
unfinished frontier positions (#316, #230) both complete within 6 hours from W2's library; and with them the whole-game
first-player strategy projects to no more than the trained net's ~660K bits (refuted when even a lower bound
exceeds it)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w3 import SPEC, w3_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w3.json.gz"
FILES = (FILE, "c49_w2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return w3_report(load_evidence(EVIDENCE / FILE), SPEC, load_evidence(EVIDENCE / FILES[1]))


def test_c49_w3_both_unfinished_frontier_positions_complete_within_6_hours():
    assert _report()["complete"]["verdict"] == "supported"


def test_c49_w3_completion_is_undecidable():
    assert _report()["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w3_the_whole_game_strategy_projects_no_larger_than_the_net():
    assert _report()["projection"]["verdict"] == "supported"


def test_c49_w3_the_projection_is_undecidable():
    assert _report()["projection"]["verdict"] in {"inconclusive", "not_run"}
