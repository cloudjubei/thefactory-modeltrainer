"""§3.6 S4 — PRE-REGISTERED claims on complete certified first-player strategies from S3's 4 ply-8 roots with
size-scored local-search leaves (a leaf may carry exceptions when >= 10x under its table), judged by
harness.floor_s4.s4_report (= floor_s2's report) from evidence/c49_s4.json.gz (scripts/c4_strategy_s2.py --spec
harness.floor_s4): at least 3 of 4 complete within 4 hours; median compression at least 10x; and on every root both
S3 (evidence/c49_s3.json.gz) and S4 complete, S4's strategy is no larger than S3's."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_s4 import SPEC, s4_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s4.json.gz"
FILES = (FILE, "c49_s3.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return s4_report(load_evidence(EVIDENCE / FILE), SPEC)


def _both_complete() -> list:
    s3 = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_s3.json.gz")["roots"]}
    s4 = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    assert set(s3) == set(s4)
    return [(s3[i], s4[i]) for i in s4 if s3[i]["complete"] and s4[i]["complete"]]


def test_c49_s4_three_of_four_ply_8_roots_get_a_complete_certified_strategy_within_4_hours():
    assert _report()["complete"]["verdict"] == "supported"


def test_c49_s4_completion_is_undecidable():
    assert _report()["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_s4_the_strategies_are_at_least_10_times_smaller_than_their_table():
    assert _report()["compression"]["verdict"] == "supported"


def test_c49_s4_compression_is_undecidable():
    assert _report()["compression"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_s4_is_no_larger_than_s3_on_every_root_both_complete():
    pairs = _both_complete()
    assert pairs and all(b["bits"]["nodes"] <= a["bits"]["nodes"] for a, b in pairs)


def test_c49_s4_no_root_completed_by_both():
    assert not _both_complete()
