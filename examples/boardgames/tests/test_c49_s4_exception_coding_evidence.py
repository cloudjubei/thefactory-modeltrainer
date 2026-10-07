"""§3.6 — PRE-REGISTERED claims on re-costing the S4 strategies with exceptions coded in walk order, from
evidence/c49_s4_exception_coding.json.gz (scripts/c4_exception_coding.py over evidence/c49_s4.json.gz): the median
completed strategy's exception bits fall by at least 40%; and with them S4's strategy for #3591 is no larger than
S3's (34,115 bits, evidence/c49_s3.json.gz)."""
from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s4_exception_coding.json.gz"
FILES = (FILE, "c49_s3.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _done() -> list:
    rows = load_evidence(EVIDENCE / FILE)["roots"]
    assert len(rows) == 4
    return [r for r in rows if r["complete"] and r["charged_bits"]]


def _saving():
    return statistics.median(1 - r["walk_bits"] / r["charged_bits"] for r in _done())


def _gap():
    s3 = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_s3.json.gz")["roots"]}[3591]
    s4 = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}[3591]
    assert s3["complete"] and s4["complete"]
    return s4["nodes_walk"] - s3["bits"]["nodes"]


def test_c49_s4_walk_order_coding_cuts_exception_bits_by_40_percent():
    assert _done() and _saving() >= 0.4


def test_c49_s4_the_coding_saving_is_undecidable():
    assert _done() and 0.1 <= _saving() < 0.4


def test_c49_s4_with_walk_order_coding_3591_is_no_larger_than_s3():
    assert _gap() <= 0
