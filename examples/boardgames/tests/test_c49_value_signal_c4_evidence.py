"""§C.49 T14 — the PRE-REGISTERED claims that two value-signal fixes make Connect-4 nets play the optimal move at
more of their own opening positions, each paired by seed against the same process without it, judged by
harness.floor_c4_value_signal.pilot_report from evidence/c49_T14_readout.json.gz (scripts/c4_pilot_readout.py over
the nets trained by scripts/c4_solver_free.py):

  tree_value  value targets for the strategy-tree positions from the relabel search's root value;
  n_step      value targets bootstrapped from a lagged target net 8 moves ahead.

Each proof passes only on SUPPORTED; each undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this
file and harness/floor_c4_value_signal.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_c4_value_signal import SPEC, pilot_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T14_readout.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _verdicts() -> dict:
    return {t: r["verdict"] for t, r in pilot_report(load_evidence(EVIDENCE / FILE), SPEC)["treatments"].items()}


def test_c49_T14_tree_value_targets_raise_the_share_of_optimal_opening_moves_on_3_of_4_seeds_by_10_points():
    assert _verdicts()["tree_value"] == "supported"


def test_c49_T14_tree_value_is_undecidable():
    assert _verdicts()["tree_value"] in {"inconclusive", "not_run"}


def test_c49_T14_n_step_value_targets_raise_the_share_of_optimal_opening_moves_on_3_of_4_seeds_by_10_points():
    assert _verdicts()["n_step"] == "supported"


def test_c49_T14_n_step_is_undecidable():
    assert _verdicts()["n_step"] in {"inconclusive", "not_run"}
