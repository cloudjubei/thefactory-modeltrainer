"""§3.6 W8 — PRE-REGISTERED claims on W6's 32 frontier positions built, improved and improved with lookahead, judged by
harness.floor_w8 from evidence/c49_w8.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_w8): at least 28 of
the 32 complete within 2 hours; the whole game projects to at most nine tenths of W7's 1,465,679 bits; it still
projects above the trained net's ~660K (the whole bootstrap interval); and lookahead cuts what plain improvement left
by at least 10% inside the 32 builds."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w8 import SPEC, w8_lookahead, w8_projection, w8_report, w8_vs_w7

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w8.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_w8_28_of_32_complete_within_2_hours():
    assert w8_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_w8_completion_is_undecidable():
    assert w8_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w8_the_whole_game_projects_under_nine_tenths_of_w7():
    assert w8_vs_w7(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w8_the_projection_against_w7_is_undecidable():
    assert w8_vs_w7(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_w8_the_whole_game_still_projects_above_the_net():
    assert w8_projection(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w8_the_projection_against_the_net_is_undecidable():
    assert w8_projection(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_w8_lookahead_cuts_a_tenth_inside_the_builds():
    assert w8_lookahead(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w8_the_lookahead_cut_is_undecidable():
    assert w8_lookahead(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
