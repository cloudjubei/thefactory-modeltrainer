"""§C.49 H2 — the PRE-REGISTERED claim that training the net beside the hybrid's exact opening table (self-play from
the table's frontier, the strategy tree walked from there) makes it play the optimal move more often one ply past the
table than the base process does, seed-matched over seven pairs, by an exact one-sided sign-flip permutation test at
5%. Judged by harness.floor_h2.h2_report from evidence/c49_H2_readout.json.gz (scripts/c4_h2_readout.py over the nets
of scripts/c4_solver_free.py --opening-table 5). The proof passes only on SUPPORTED; the undecidable proof passes on
INCONCLUSIVE or NOT_RUN. The register pins this file and harness/floor_h2.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_h2 import SPEC, h2_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_H2_readout.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _verdict() -> str:
    return h2_report(load_evidence(EVIDENCE / FILE), SPEC)["verdict"]


def test_c49_H2_the_table_trained_net_plays_the_optimal_move_more_often_one_ply_past_the_table():
    assert _verdict() == "supported"


def test_c49_H2_is_undecidable():
    assert _verdict() in {"inconclusive", "not_run"}
