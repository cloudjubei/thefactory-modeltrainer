"""§3.6 W1 — PRE-REGISTERED claims on the whole-game first-player strategy, projected, judged by
harness.floor_w1.w1_report from evidence/c49_w1.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_w1): at least 6
of 8 sampled frontier positions of the ply-8 opening get a complete certified strategy within 2 hours; and the
projected whole strategy (opening + 671 frontier strategies at the sample's mean size) is no larger than the trained
net (660K bits) — refuted when even its lower bound exceeds 6.6M bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w1 import SPEC, w1_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return w1_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_w1_six_of_eight_frontier_positions_get_a_complete_strategy_within_2_hours():
    assert _report()["complete"]["verdict"] == "supported"


def test_c49_w1_completion_is_undecidable():
    assert _report()["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w1_the_projected_whole_game_strategy_is_no_larger_than_the_net():
    assert _report()["projection"]["verdict"] == "supported"


def test_c49_w1_the_projection_is_undecidable():
    assert _report()["projection"]["verdict"] in {"inconclusive", "not_run"}
