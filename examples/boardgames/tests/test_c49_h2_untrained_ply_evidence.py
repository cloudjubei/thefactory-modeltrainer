"""§C.49 H2b — DESCRIPTIVE, registered after the data: the table-trained nets also err less at ply 8, which their
training never walked. Read from evidence/c49_H2_readout.json.gz through harness.floor_h2.h2_report. The register
pins this file."""
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
def _report() -> dict:
    return h2_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_H2b_the_table_trained_hybrids_fail_at_fewer_ply_8_positions_than_base_and_none_certifies():
    r = _report()
    assert r["failures_by_ply"] == {"base": {"6": 496, "8": 865}, "table": {"6": 366, "8": 681}}
    assert r["certified"] == {"base": 0, "table": 0}
