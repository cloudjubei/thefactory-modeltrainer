"""§3.6 R1 — PRE-REGISTERED claim on F2's leaf references coded with first_use (harness.reference_coding), judged by
harness.floor_r1 from evidence/c49_r1.json.gz (scripts/c4_reference_coding.py --spec harness.floor_r1): the code
saves at least a tenth of F2's whole 455,959 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_r1 import SPEC, r1_saving

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_r1.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_r1_first_use_references_save_a_tenth_of_the_whole_game():
    assert r1_saving(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_r1_the_saving_is_undecidable():
    assert r1_saving(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
