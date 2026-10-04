"""§C.49 D3 — the PRE-REGISTERED claim that the T10 nets' VALUE HEADS back their own wrong opening moves, judged by
harness.value_diagnosis.d3_report on D3_SPEC from evidence/c49_D3_value_head.json.gz
(scripts/c4_value_diagnosis.py). The proof passes only on SUPPORTED; the undecidable proof passes on INCONCLUSIVE or
NOT_RUN. The register pins this file and harness/value_diagnosis.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.value_diagnosis import D3_SPEC, d3_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_D3_value_head.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return d3_report(load_evidence(EVIDENCE / FILE), D3_SPEC)


def test_c49_D3_the_value_head_rates_the_net_s_wrong_move_at_least_as_high_as_the_best_move_in_60pct_of_errors():
    assert _report()["verdict"] == "supported"


def test_c49_D3_is_undecidable():
    assert _report()["verdict"] in {"inconclusive", "not_run"}
