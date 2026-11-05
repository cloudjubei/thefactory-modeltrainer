"""§3.6 R2 — PRE-REGISTERED claims on G1 rebuilt with the builder counting its references by first_use, as the judges
do, judged by harness.floor_r2 from evidence/c49_r2.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_r2)
against G1 (evidence/c49_g1.json.gz) and B1's control (evidence/c49_b1_control.json.gz), both coded by first_use with
each library map a leaf plays paid once: at least 14 of the 16 positions complete within 2 hours, and the total is no
more than nine tenths of G1's 74,556 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_r2 import SPEC, r2_report, r2_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_r2.json.gz"
FILES = (FILE, "c49_g1.json.gz", "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=3)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return r2_size(_evidence(), _evidence("c49_g1.json.gz"), _evidence("c49_b1_control.json.gz"), SPEC)


def test_c49_r2_14_of_16_complete_counting_references_by_first_use():
    assert r2_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_r2_completion_is_undecidable():
    assert r2_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_r2_counting_references_as_judged_makes_the_heavy_tail_a_tenth_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_r2_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
