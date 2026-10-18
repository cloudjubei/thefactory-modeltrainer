"""§3.6 F2 — PRE-REGISTERED claims on the whole game rebuilt with L5's rule (the safe moves on unlevelled cells a last
level): every one of the opening's 671 frontier positions in F1's order and waves from the same seeded library, judged
by harness.floor_f2 from evidence/c49_f2.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_f2): every position
completes within 2 hours, and the certified whole-game strategy takes no more than nine tenths of F1's 548,424 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_f2 import SPEC, f2_complete, f2_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_f2.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_f2_every_frontier_position_completes():
    assert f2_complete(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_f2_completion_is_undecidable():
    assert f2_complete(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_f2_the_whole_strategy_is_a_tenth_smaller_than_f1_s():
    assert f2_size(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_f2_the_whole_strategy_s_size_is_undecidable():
    assert f2_size(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
