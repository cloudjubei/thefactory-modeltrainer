"""§C.49 D4b — DESCRIPTIVE, registered after the data: where D4's labels go wrong. Read from
evidence/c49_D4_labels.json.gz through harness.label_diagnosis.d4_report. The register pins this file."""
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


def test_c49_D4b_in_the_opening_the_search_mislabels_most_policy_side_errors():
    by_ply = _report()["by_ply"]
    opening = [by_ply[p] for p in ("2", "4")]
    assert "0" not in by_ply
    assert (sum(r["label_optimal"] for r in opening), sum(r["policy_side"] for r in opening)) == (12, 78)
    assert (by_ply["8"]["label_optimal"], by_ply["8"]["policy_side"]) == (1818, 3674)


def test_c49_D4b_where_the_value_head_backs_the_error_the_search_follows_it():
    assert _report()["value_side"] == {"errors": 3130, "label_optimal": 559}


def test_c49_D4b_few_policy_side_errors_lie_on_the_nets_own_trees():
    assert _report()["by_tree"]["on_tree"] == {"policy_side": 84, "label_optimal": 27}
