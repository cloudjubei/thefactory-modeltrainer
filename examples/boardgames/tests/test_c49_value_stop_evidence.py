"""§C.49 T11 — the PRE-REGISTERED claim that the value-aware stop signal stays safe and timely on tic-tac-toe under a
weakened (50-simulation) relabel search, judged by harness.floor_value_stop.value_stop_report. The proof passes only
on SUPPORTED; the undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file,
harness/floor_value_stop.py, harness/floor_tree.py and harness/floor_stop.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_value_stop import T11_ARM, value_stop_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = f"c49_T11_{T11_ARM}.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return value_stop_report(load_evidence(EVIDENCE / FILE))


def test_c49_T11_under_a_weak_search_the_value_stop_never_stops_early_and_stops_in_time_on_8_of_10_seeds():
    assert _report()["stop"]["verdict"] == "supported"


def test_c49_T11_the_value_stop_reading_is_undecidable():
    assert _report()["stop"]["verdict"] in {"inconclusive", "not_run"}
