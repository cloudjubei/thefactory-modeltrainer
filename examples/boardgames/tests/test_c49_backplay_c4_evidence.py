"""§C.49 T16 — the PRE-REGISTERED claim that the backward curriculum makes Connect-4 nets play the optimal move at
more of their own opening positions, paired by seed against the same process without it, judged by
harness.floor_c4_value_signal.pilot_report on harness.floor_c4_backplay.SPEC from evidence/c49_T16_readout.json.gz
(scripts/c4_pilot_readout.py --spec floor_c4_backplay --prefix c49_T16). The proof passes only on SUPPORTED; the
undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file, harness/floor_c4_backplay.py and
harness/floor_c4_value_signal.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_c4_backplay import SPEC
from harness.floor_c4_value_signal import pilot_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T16_readout.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _verdict() -> str:
    return pilot_report(load_evidence(EVIDENCE / FILE), SPEC)["treatments"]["backplay"]["verdict"]


def test_c49_T16_the_backward_curriculum_raises_the_share_of_optimal_opening_moves_on_3_of_4_seeds_by_10_points():
    assert _verdict() == "supported"


def test_c49_T16_is_undecidable():
    assert _verdict() in {"inconclusive", "not_run"}
