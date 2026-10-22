"""§3.6 G1 — PRE-REGISTERED claims on F2's 16 heaviest positions built with L5's rule and pipeline from every map B1's
control found on them, all within reach, judged by harness.floor_g1 from evidence/c49_g1.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_g1) against B1's control (evidence/c49_b1_control.json.gz), both coded
by first_use with each library map a leaf plays paid once: at least 14 of the 16 complete within 2 hours, and they take
no more than nine tenths of the control's bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g1 import SPEC, g1_report, g1_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_g1.json.gz"
FILES = (FILE, "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return g1_size(_evidence(), _evidence("c49_b1_control.json.gz"))


def test_c49_g1_14_of_16_complete_within_2_hours_with_every_map_in_reach():
    assert g1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_g1_completion_is_undecidable():
    assert g1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_g1_shared_maps_make_the_heavy_strategies_a_tenth_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_g1_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
