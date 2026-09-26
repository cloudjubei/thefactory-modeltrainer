"""§C.48 T5 — the PRE-REGISTERED claims that solver-free coverage levers take T4's fixed process to a raw-perfect
tic-tac-toe net (one claim per arm), and that the failures left are concentrated on positions the net was never
trained on (the mechanism, pooled over every arm). Judged by harness.floor_coverage.coverage_report from the five arm
files, T4's augment arm rebuilt under the T5 code, and T4's own augment evidence. Each claim's proof passes only on
SUPPORTED; its undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file and
harness/floor_coverage.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_coverage import T5_ARMS, coverage_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
ARM_FILES = {name: f"c48_T5_{name}.json.gz" for name in T5_ARMS}
FILES = (*ARM_FILES.values(), "c48_T5_repro.json.gz", "c48_T4_augment.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return coverage_report({name: load_evidence(EVIDENCE / f) for name, f in ARM_FILES.items()},
                           load_evidence(EVIDENCE / "c48_T5_repro.json.gz"),
                           load_evidence(EVIDENCE / "c48_T4_augment.json.gz"))


@pytest.mark.parametrize("arm", sorted(T5_ARMS))
def test_c48_T5_the_coverage_lever_gives_a_raw_perfect_net_on_8_of_10_seeds(arm):
    assert _report()[arm]["verdict"] == "supported"


@pytest.mark.parametrize("arm", sorted(T5_ARMS))
def test_c48_T5_arm_is_undecidable(arm):
    assert _report()[arm]["verdict"] in {"inconclusive", "not_run"}


def test_c48_T5_the_failures_left_concentrate_on_positions_the_net_was_never_trained_on():
    assert _report()["mechanism"]["verdict"] == "supported"


def test_c48_T5_mechanism_is_undecidable():
    assert _report()["mechanism"]["verdict"] in {"inconclusive", "not_run"}
