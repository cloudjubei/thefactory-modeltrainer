"""§3.6 N1 — PRE-REGISTERED claim on a learned default paid once, judged by harness.floor_n1 from
evidence/c49_n1.json.gz (scripts/c4_learned_default.py --spec harness.floor_n1): on D1's 4,523 positions the
quantized net of at most 20,000 bits misses the win at no more than 100, half the tactics order's 200."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_n1 import n1_misses

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_n1.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_n1_a_20k_bit_net_misses_half_as_often_as_tactics():
    assert n1_misses(_evidence())["verdict"] == "supported"


def test_c49_n1_the_misses_are_undecidable():
    assert n1_misses(_evidence())["verdict"] in {"inconclusive", "not_run"}
