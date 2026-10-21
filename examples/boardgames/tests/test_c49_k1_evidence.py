"""§3.6 K1 — PRE-REGISTERED claims on L4's 32 positions (F1's 16 heaviest, 16 spread over the rest) built with L6's rule
and pipeline allowing two-phase maps, judged by harness.floor_k1 from evidence/c49_k1.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_k1) against L6 (evidence/c49_l6.json.gz): at least 28 of the 32
complete within 2 hours, they take no more than nine tenths of L6's 102,289 bits, and F1's 16 heaviest of them no more
than nine tenths of L6's 94,250 on the same 16."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_k1 import SPEC, k1_heavy, k1_report, k1_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_k1.json.gz"
FILES = (FILE, "c49_l6.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return k1_size(_evidence(), _evidence("c49_l6.json.gz"), SPEC)


def _heavy() -> dict:
    return k1_heavy(_evidence(), _evidence("c49_l6.json.gz"), SPEC)


def test_c49_k1_28_of_32_complete_within_2_hours_with_two_phase_maps():
    assert k1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_k1_completion_is_undecidable():
    assert k1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_k1_two_phase_maps_make_the_strategies_a_tenth_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_k1_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}


def test_c49_k1_two_phase_maps_make_the_heavy_sixteen_a_tenth_smaller():
    assert _heavy()["verdict"] == "supported"


def test_c49_k1_the_heavy_size_is_undecidable():
    assert _heavy()["verdict"] in {"inconclusive", "not_run"}
