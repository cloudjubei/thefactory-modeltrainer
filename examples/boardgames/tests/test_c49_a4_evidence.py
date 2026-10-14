"""§3.6 A4 — PRE-REGISTERED claims on eight frontier positions (#95, #271, #374, #512 — those W7 left stuck — and
#594, #535, #98, #457) built as W6, improved as W7, then improved again with lookahead, judged by harness.floor_a4
from evidence/c49_a4.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_a4): lookahead cuts the total that plain
improvement left by at least 15%, and it cuts at least half the heavy positions plain improvement could not change."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_a4 import SPEC, a4_further, a4_unstuck

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a4.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_a4_lookahead_cuts_what_plain_improvement_left_by_15_percent():
    assert a4_further(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_a4_the_further_cut_is_undecidable():
    assert a4_further(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_a4_lookahead_unsticks_at_least_half_the_stuck_positions():
    assert a4_unstuck(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_a4_unsticking_is_undecidable():
    assert a4_unstuck(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
