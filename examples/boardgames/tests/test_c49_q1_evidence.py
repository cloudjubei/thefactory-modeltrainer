"""§3.6 Q1 — PRE-REGISTERED claims on a seeded sample of 60 of F2's leaves with exceptions, judged by harness.floor_q1
from evidence/c49_q1.json.gz (scripts/c4_exception_classes.py --spec harness.floor_q1): flags coded per class, and
moves coded tactics first, each save at least a tenth of the leaves' coded exception bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_q1 import q1_flags, q1_moves

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_q1.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_q1_flags_coded_per_class_save_a_tenth_of_the_exception_bits():
    assert q1_flags(_evidence())["verdict"] == "supported"


def test_c49_q1_the_flags_saving_is_undecidable():
    assert q1_flags(_evidence())["verdict"] in {"inconclusive", "not_run"}


def test_c49_q1_moves_coded_tactics_first_save_a_tenth_of_the_exception_bits():
    assert q1_moves(_evidence())["verdict"] == "supported"


def test_c49_q1_the_moves_saving_is_undecidable():
    assert q1_moves(_evidence())["verdict"] in {"inconclusive", "not_run"}
