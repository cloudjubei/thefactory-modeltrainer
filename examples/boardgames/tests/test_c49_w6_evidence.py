"""§3.6 W6 — PRE-REGISTERED claims on the next 32 of W1's seeded frontier positions, built exactly as W5b built its 8
without a library, judged by harness.floor_w6 from evidence/c49_w6.json.gz (scripts/c4_strategy_s2.py --spec
harness.floor_w6): at least 28 of the 32 complete within 2 hours; the whole game projects above the trained net's
~660K bits (the whole bootstrap interval); and the 4 largest of the 32 strategies carry at least half their bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w6 import SPEC, w6_projection, w6_report, w6_tail

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w6.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_w6_28_of_32_complete_within_2_hours():
    assert w6_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_w6_completion_is_undecidable():
    assert w6_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w6_the_whole_game_projects_above_the_net():
    assert w6_projection(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w6_the_projection_against_the_net_is_undecidable():
    assert w6_projection(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_w6_the_4_largest_carry_half_the_bits():
    assert w6_tail(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w6_the_tail_is_undecidable():
    assert w6_tail(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
