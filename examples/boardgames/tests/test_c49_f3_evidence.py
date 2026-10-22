"""§3.6 F3 — PRE-REGISTERED claims on the whole game rebuilt with L6's rule (a deciding level's ties broken by tactics):
every one of the opening's 671 frontier positions in F1's order and waves from the same seeded library, judged by
harness.floor_f3 from evidence/c49_f3.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_f3): every position
completes within 2 hours, and the certified whole-game strategy takes no more than nine tenths of F2's 455,959 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_f3 import SPEC, f3_complete, f3_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_f3.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_f3_every_frontier_position_completes():
    assert f3_complete(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_f3_completion_is_undecidable():
    assert f3_complete(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_f3_the_whole_strategy_is_a_tenth_smaller_than_f2_s():
    assert f3_size(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_f3_the_whole_strategy_s_size_is_undecidable():
    assert f3_size(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
