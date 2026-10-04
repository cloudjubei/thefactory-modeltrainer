"""§C.49 H1 — the PRE-REGISTERED claims on how far a self-play net carries play beyond an exact opening exception
table: with the table covering the first player's positions before ply 3 (h129) or ply 5 (h130), the hybrid is
certified one White ply further (through 5 or 7 plies) on at least 8 of the 10 base-recipe nets. Judged by
harness.floor_hybrid.hybrid_report from evidence/c49_H1_hybrid.json.gz (scripts/c4_hybrid.py). Each proof passes only
on SUPPORTED; each undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file and
harness/floor_hybrid.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_hybrid import SPEC, hybrid_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_H1_hybrid.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _verdicts() -> dict:
    return {h: r["verdict"] for h, r in hybrid_report(load_evidence(EVIDENCE / FILE), SPEC)["horizons"].items()
            if "verdict" in r}


def test_c49_H1_with_a_table_before_ply_3_the_net_carries_ply_4_on_8_of_10_nets():
    assert _verdicts()["3"] == "supported"


def test_c49_H1_horizon_3_is_undecidable():
    assert _verdicts()["3"] in {"inconclusive", "not_run"}


def test_c49_H1_with_a_table_before_ply_5_the_net_carries_ply_6_on_8_of_10_nets():
    assert _verdicts()["5"] == "supported"


def test_c49_H1_horizon_5_is_undecidable():
    assert _verdicts()["5"] in {"inconclusive", "not_run"}
