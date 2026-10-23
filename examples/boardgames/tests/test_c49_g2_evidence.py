"""§3.6 G2 — PRE-REGISTERED claims on the whole game built again with L5's rule and pipeline from every map F2 found,
all within reach, no waves, judged by harness.floor_g2 from evidence/c49_g2.json.gz (scripts/c4_strategy_s2.py --spec
harness.floor_g2) against F2 (evidence/c49_f2.json.gz), both coded by first_use with each library map a leaf plays paid
once: every position completes within 2 hours, and the whole game takes no more than nine tenths of F2's 398,195."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g2 import SPEC, g2_complete, g2_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_g2.json.gz"
FILES = (FILE, "c49_f2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return g2_size(_evidence(), _evidence("c49_f2.json.gz"), SPEC)


def test_c49_g2_every_frontier_position_completes_from_f2_s_maps():
    assert g2_complete(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_g2_completion_is_undecidable():
    assert g2_complete(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_g2_a_second_pass_makes_the_whole_game_a_tenth_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_g2_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
