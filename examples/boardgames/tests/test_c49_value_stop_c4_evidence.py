"""§C.49 T12 — the PRE-REGISTERED claim that the solver-free Connect-4 process, stopped by the value-aware signal,
stops on a net the solver certifies through 10 plies on at least 8 of 10 seeds, judged by harness.floor_c4.c4_report
on harness.floor_c4_value.SPEC. The proof passes only on SUPPORTED; the undecidable proof passes on INCONCLUSIVE or
NOT_RUN. The register pins this file, harness/floor_c4_value.py and harness/floor_c4.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_c4 import c4_report
from harness.floor_c4_value import SPEC

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T12_value_stop.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return c4_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_T12_the_value_stopped_solver_free_run_ends_on_a_net_certified_through_10_plies_on_8_of_10_seeds():
    assert _report()["verdict"] == "supported"


def test_c49_T12_is_undecidable():
    assert _report()["verdict"] in {"inconclusive", "not_run"}
