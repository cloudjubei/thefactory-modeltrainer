"""§3.6 A3 — PRE-REGISTERED claims on A1's four heavy frontier positions (#95, #565, #594, #535) built exactly as W6
built them and then improved for the rest of the 2 hours (the costliest leaves with exceptions split first, a split
kept only if the strategy shrinks), judged by harness.floor_a3 from evidence/c49_a3.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_a3): all four complete within 2 hours, and together they take no more
than the smaller build per position of W6 and A1 (28,248 bits)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_a3 import SPEC, a3_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a3.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_a3_all_four_complete_within_2_hours():
    assert a3_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_a3_completion_is_undecidable():
    assert a3_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_a3_improving_anytime_matches_the_best_build_per_position():
    assert a3_report(_evidence(), SPEC)["size"]["verdict"] == "supported"


def test_c49_a3_the_size_is_undecidable():
    assert a3_report(_evidence(), SPEC)["size"]["verdict"] in {"inconclusive", "not_run"}
