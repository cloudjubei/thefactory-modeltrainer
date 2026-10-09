"""§3.6 W5 — PRE-REGISTERED claims on W2 rebuilt with the root-only long search P2 meant (budgets [[0, 1200], [2, 30]])
and the C walk, judged by harness.floor_w5 from evidence/c49_w5.json.gz (scripts/c4_strategy_s2.py --spec
harness.floor_w5): at least 6 of the 8 complete within 2 hours; the whole game projects to no more than W3's 1.62M-bit
lower bound (refuted when even W5's lower bound exceeds it); and the 8 builds take at most half W2's total time."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w5 import SPEC, time_report, w5_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w5.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_w5_six_of_eight_complete_within_2_hours():
    assert w5_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_w5_completion_is_undecidable():
    assert w5_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w5_the_whole_game_projects_within_w3_s_bound():
    assert w5_report(_evidence(), SPEC)["projection"]["verdict"] == "supported"


def test_c49_w5_the_projection_is_undecidable():
    assert w5_report(_evidence(), SPEC)["projection"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w5_the_builds_take_half_w2_s_time():
    assert time_report(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w5_the_time_change_is_undecidable():
    assert time_report(_evidence(), SPEC)["verdict"] == "inconclusive"
