"""§C.49 T10 — the PRE-REGISTERED claim that the solver-free process reaches perfect play from the start on
Connect-4 through 10 plies: trained with the solver forbidden, stopped by its own walk reading full agreement, and
only then certified by the solver, judged by harness.floor_c4.c4_report. The claim's proof passes only on
SUPPORTED; its undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file and
harness/floor_c4.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_c4 import c4_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T10_solver_free.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return c4_report(load_evidence(EVIDENCE / FILE))


def test_c49_T10_the_solver_free_process_stops_on_a_net_certified_through_10_plies_on_8_of_10_seeds():
    assert _report()["verdict"] == "supported"


def test_c49_T10_is_undecidable():
    assert _report()["verdict"] in {"inconclusive", "not_run"}
