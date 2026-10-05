"""§C.49 H3 — the PRE-REGISTERED claims of training the plies the net must carry: H3 (the strategy tree walked from
the opening table's frontier through plies 6 and 8) against H2's table-arm nets of the same seeds — (h134) a higher
share of optimal moves at the fixed ply-8 set, by an exact one-sided sign-flip permutation test at 5%; (h135) its
ply-6 share still rising at the end of training (passes 16-20 against 11-15). Judged by harness.floor_h3.h3_report
from evidence/c49_H3_readout.json.gz (scripts/c4_h3_readout.py). Each proof passes only on SUPPORTED; each undecidable
proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file and harness/floor_h3.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_h3 import SPEC, h3_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_H3_readout.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return h3_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_H3_training_plies_6_and_8_raises_the_ply_8_share_over_h2():
    assert _report()["s8"]["verdict"] == "supported"


def test_c49_H3_the_ply_8_reading_is_undecidable():
    assert _report()["s8"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_H3_the_ply_6_share_is_still_rising_at_iteration_20():
    assert _report()["curve"]["verdict"] == "supported"


def test_c49_H3_the_curve_reading_is_undecidable():
    assert _report()["curve"]["verdict"] in {"inconclusive", "not_run"}
