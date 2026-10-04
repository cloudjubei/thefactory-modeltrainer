"""§C.49 D4 — the PRE-REGISTERED claim that at the T10 nets' POLICY-SIDE errors their own 200-sim relabel search
labels an optimal move, judged by harness.label_diagnosis.d4_report on D4_SPEC from evidence/c49_D4_labels.json.gz
(scripts/c4_label_diagnosis.py). The proof passes only on SUPPORTED; the undecidable proof passes on INCONCLUSIVE or
NOT_RUN. The register pins this file, harness/label_diagnosis.py and harness/value_diagnosis.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.label_diagnosis import D4_SPEC, d4_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_D4_labels.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return d4_report(load_evidence(EVIDENCE / FILE), D4_SPEC)


def test_c49_D4_at_policy_side_errors_the_relabel_search_labels_an_optimal_move_in_60pct():
    assert _report()["verdict"] == "supported"


def test_c49_D4_is_undecidable():
    assert _report()["verdict"] in {"inconclusive", "not_run"}
