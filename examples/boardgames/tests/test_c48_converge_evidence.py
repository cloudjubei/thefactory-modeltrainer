"""§C.48 T2 — the PRE-REGISTERED claims that the generic tic-tac-toe process, run to convergence (30 iterations),
gives a RAW net optimal at every raw position, judged by harness.floor_converge.converge_report from the two arm
files. Each claim's proof passes only on SUPPORTED; its undecidable proof passes on INCONCLUSIVE or NOT_RUN. The
register pins this file and harness/floor_converge.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_converge import converge_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c48_T2_augment.json.gz", "c48_T2_no_augment.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return converge_report({"augment": load_evidence(EVIDENCE / FILES[0]),
                            "no_augment": load_evidence(EVIDENCE / FILES[1])})


def test_c48_T2_run_to_convergence_the_raw_net_is_perfect_at_every_position_with_augmentation():
    assert _report()["augment"]["verdict"] == "supported"


def test_c48_T2_augment_is_undecidable():
    assert _report()["augment"]["verdict"] in {"inconclusive", "not_run"}


def test_c48_T2_run_to_convergence_the_raw_net_is_perfect_at_every_position_WITHOUT_any_symmetry():
    assert _report()["no_augment"]["verdict"] == "supported"


def test_c48_T2_no_augment_is_undecidable():
    assert _report()["no_augment"]["verdict"] in {"inconclusive", "not_run"}
