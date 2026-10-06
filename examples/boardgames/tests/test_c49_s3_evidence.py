"""§3.6 S3 — PRE-REGISTERED claims on complete certified first-player strategies from 4 ply-8 roots of the label cache
(leaves searched only from ply 12), judged by harness.floor_s3.s3_report (= floor_s2's report) from
evidence/c49_s3.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_s3): at least 3 of 4 complete within 4 hours;
median compression at least 10x against the same strategy as a 3-bit table. The register pins this file and
harness/floor_s3.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_s3 import SPEC, s3_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s3.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return s3_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_s3_three_of_four_ply_8_roots_get_a_complete_certified_strategy_within_4_hours():
    assert _report()["complete"]["verdict"] == "supported"


def test_c49_s3_completion_is_undecidable():
    assert _report()["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_s3_the_strategies_are_at_least_10_times_smaller_than_their_table():
    assert _report()["compression"]["verdict"] == "supported"


def test_c49_s3_compression_is_undecidable():
    assert _report()["compression"]["verdict"] in {"inconclusive", "not_run"}
