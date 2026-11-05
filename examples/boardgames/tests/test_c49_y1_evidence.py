"""§3.6 Y1 — PRE-REGISTERED claims on WeakC4 (2swap) as a yardstick for G3, judged by harness.floor_y1 from
evidence/c49_weakc4.json.gz (scripts/c4_weakc4_yardstick.py --weakc4 <checkout>): our "exact" rule with all ten levels
wins with WeakC4's diagram at nine tenths or more of its leaves, and WeakC4's whole solution priced in our code takes
no more than a third of G3's 280,640 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_y1 import SPEC, y1_language, y1_price

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_weakc4.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_y1_our_exact_rule_plays_weakc4_s_diagrams():
    assert y1_language(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_y1_the_language_is_undecidable():
    assert y1_language(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_y1_weakc4_in_our_code_is_a_third_of_g3_or_less():
    assert y1_price(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_y1_the_price_is_undecidable():
    assert y1_price(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
