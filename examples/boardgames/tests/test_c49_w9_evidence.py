"""§3.6 W9 — PRE-REGISTERED claims on the next 32 frontier positions of W1's seeded order (never built), with L2's rule
(the first level holding any safe move decides) and pipeline, judged by harness.floor_w9 from evidence/c49_w9.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_w9): at least 28 of the 32 complete within 2 hours, and the whole
game projects below the trained net's ~660K bits (the whole bootstrap interval, every root finished)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w9 import SPEC, w9_below, w9_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w9.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_w9_28_of_32_complete_within_2_hours():
    assert w9_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_w9_completion_is_undecidable():
    assert w9_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w9_the_whole_game_projects_below_the_net():
    assert w9_below(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w9_the_projection_against_the_net_is_undecidable():
    assert w9_below(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
