"""§3.6 W2 — PRE-REGISTERED claims on the whole-game projection redone (root searches, honest budgets, walk-order
exceptions, a map library shared across two waves), judged by harness.floor_w2 from evidence/c49_w2.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_w2): at least 6 of W1's 8 sampled frontier positions complete within 2
hours; the projected whole first-player strategy is at most half of W1's 7.97M bits (refuted when even its lower bound
exceeds W1's); and at least a tenth of the second wave's leaves use a map from the first wave's library."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w2 import SPEC, library_report, w2_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w2.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def _report() -> dict:
    return w2_report(_evidence(), SPEC)


def test_c49_w2_six_of_eight_frontier_positions_complete_within_2_hours():
    assert _report()["complete"]["verdict"] == "supported"


def test_c49_w2_completion_is_undecidable():
    assert _report()["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w2_the_projected_whole_game_strategy_is_half_of_w1_s_or_less():
    assert _report()["projection"]["verdict"] == "supported"


def test_c49_w2_the_projection_is_undecidable():
    assert _report()["projection"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w2_a_tenth_of_the_second_wave_s_leaves_use_the_library():
    assert library_report(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w2_library_use_is_undecidable():
    assert library_report(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
