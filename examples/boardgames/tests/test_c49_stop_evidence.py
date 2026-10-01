"""§C.49 T8 — the PRE-REGISTERED claim that the solver-free stop signal (the strategy-tree walk's disagreement count
reaching zero) never stops on a net whose own first-player tree the solver does not certify, and stops within the
registered latency of certification on enough seeds, judged by harness.floor_stop.stop_report. The claim's proof
passes only on SUPPORTED; its undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file,
harness/floor_stop.py and harness/arm_judge.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_stop import T8_ARM, stop_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = f"c49_T8_{T8_ARM}.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return stop_report(load_evidence(EVIDENCE / FILE))


def test_c49_T8_the_stop_signal_never_stops_early_and_stops_in_time_on_8_of_10_seeds():
    assert _report()["verdict"] == "supported"


def test_c49_T8_the_stop_signal_is_undecidable():
    assert _report()["verdict"] in {"inconclusive", "not_run"}
