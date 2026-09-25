"""§C.47 leg F — the PRE-REGISTERED floor claims, judged by harness.floor.floor_report from the arm evidence
(evidence/c47_F.json.gz, R200S's recipe on seeds 81-100) and the bit-for-bit rebuild of R200S seed 41 under the same
code (evidence/c47_F_repro.json.gz). Each claim has a proof (passes only on SUPPORTED) and an undecidable proof
(passes on INCONCLUSIVE or NOT_RUN); a claim reads REFUTED only when both fail. The register pins this file and
harness/floor.py, so neither the bars nor these assertions can move after the data exists."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor import floor_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c47_F.json.gz", "c47_F_repro.json.gz")
UNDECIDABLE = {"inconclusive", "not_run"}
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return floor_report(load_evidence(EVIDENCE / FILES[0]), load_evidence(EVIDENCE / FILES[1]))


def test_c47_F1_the_floor_is_met_under_the_symmetry_averaged_policy():
    assert _report()["F1"]["verdict"] == "supported"


def test_c47_F1_is_undecidable():
    assert _report()["F1"]["verdict"] in UNDECIDABLE


def test_c47_F2_the_averaged_operator_is_perfect_on_more_seeds_than_the_strict_raw_reading():
    assert _report()["F2"]["verdict"] == "supported"


def test_c47_F2_is_undecidable():
    assert _report()["F2"]["verdict"] in UNDECIDABLE
