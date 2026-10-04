"""§C.49 T17 — the PRE-REGISTERED claims of the powered Connect-4 comparison: each of three value-signal fixes raises
the share of optimal moves on one fixed set of 5,142 exactly valued positions, against base on seven seed-matched
pairs, by an exact one-sided sign-flip permutation test with Holm's correction across the three. Judged by
harness.floor_c4_powered.powered_report from evidence/c49_T17_fixed_set.json.gz (scripts/c4_fixed_set_readout.py
--runs c49_T17). Each proof passes only on SUPPORTED; each undecidable proof passes on INCONCLUSIVE or NOT_RUN. The
register pins this file and harness/floor_c4_powered.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_c4_powered import SPEC, powered_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T17_fixed_set.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _verdicts() -> dict:
    return {t: r["verdict"] for t, r in powered_report(load_evidence(EVIDENCE / FILE), SPEC)["treatments"].items()}


def test_c49_T17_n_step_value_targets_raise_the_fixed_set_share_against_base():
    assert _verdicts()["n_step"] == "supported"


def test_c49_T17_n_step_is_undecidable():
    assert _verdicts()["n_step"] in {"inconclusive", "not_run"}


def test_c49_T17_tree_value_targets_raise_the_fixed_set_share_against_base():
    assert _verdicts()["tree_value"] == "supported"


def test_c49_T17_tree_value_is_undecidable():
    assert _verdicts()["tree_value"] in {"inconclusive", "not_run"}


def test_c49_T17_the_backward_curriculum_raises_the_fixed_set_share_against_base():
    assert _verdicts()["backplay"] == "supported"


def test_c49_T17_backplay_is_undecidable():
    assert _verdicts()["backplay"] in {"inconclusive", "not_run"}
