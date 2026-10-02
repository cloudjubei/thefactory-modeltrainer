"""§C.49 D2 — the PRE-REGISTERED claim that enough search makes the Connect-4 opening labels right, judged by
harness.floor_opening.opening_report on the six saved T10 nets. The claim's proof passes only on SUPPORTED; its
undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file and harness/floor_opening.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_opening import opening_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_D2_opening.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return opening_report(load_evidence(EVIDENCE / FILE))


def test_c49_D2_some_search_budget_prefers_an_optimal_opening_move_at_95_percent_of_every_net_s_positions():
    assert _report()["verdict"] == "supported"


def test_c49_D2_is_undecidable():
    assert _report()["verdict"] in {"inconclusive", "not_run"}
