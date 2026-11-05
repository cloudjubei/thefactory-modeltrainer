"""§3.6 N2 — PRE-REGISTERED claims on L4's 32 positions (held out of N1's training) built with L5's rule and pipeline
breaking a deciding level's ties by N1's 19,996-bit net, judged by harness.floor_n2 from evidence/c49_n2.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_n2) against L6 (evidence/c49_l6.json.gz, ties by tactics): at least
28 of the 32 complete within 2 hours, and they take no more than nine tenths of L6's 102,289 bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_n2 import SPEC, n2_report, n2_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_n2.json.gz"
FILES = (FILE, "c49_l6.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return n2_size(_evidence(), _evidence("c49_l6.json.gz"), SPEC)


def test_c49_n2_28_of_32_complete_within_2_hours_breaking_ties_by_the_net():
    assert n2_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_n2_completion_is_undecidable():
    assert n2_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_n2_breaking_ties_by_the_net_makes_the_strategies_a_tenth_smaller_than_tactics():
    assert _size()["verdict"] == "supported"


def test_c49_n2_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
