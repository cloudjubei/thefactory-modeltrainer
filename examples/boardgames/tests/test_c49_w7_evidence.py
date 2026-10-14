"""§3.6 W7 — PRE-REGISTERED claims on W6's 32 frontier positions rebuilt exactly as W6 built them plus the builder's
anytime improvement, judged by harness.floor_w7 from evidence/c49_w7.json.gz (scripts/c4_strategy_s2.py --spec
harness.floor_w7): at least 28 of the 32 complete within 2 hours; the whole game projects to at most half W6's
4,306,421 bits; and it still projects above the trained net's ~660K (the whole bootstrap interval)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w7 import SPEC, w7_halving, w7_projection, w7_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w7.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_w7_28_of_32_complete_within_2_hours():
    assert w7_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_w7_completion_is_undecidable():
    assert w7_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w7_improving_halves_the_whole_game():
    assert w7_halving(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w7_the_halving_is_undecidable():
    assert w7_halving(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_w7_the_whole_game_still_projects_above_the_net():
    assert w7_projection(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w7_the_projection_against_the_net_is_undecidable():
    assert w7_projection(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
