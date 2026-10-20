"""§3.6 D1 — claim registered before the measured run (after a 621-position smoke pilot that did not score "tactics"),
judged by harness.floor_d1 from evidence/c49_d1.json.gz (scripts/c4_default_accuracy.py --spec harness.floor_l5
--targets c49_l5.json.gz): on positions sampled from L5's certified leaves, the "tactics" default — the lowest after
preferring moves that win by force in two, then moves that do not lose by force in two — misses at most two thirds as
many positions as the lowest."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_d1 import SPEC, d1_misses

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_d1.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_d1_the_tactics_default_misses_two_thirds_of_the_lowest_s_positions_or_fewer():
    assert d1_misses(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_d1_the_misses_are_undecidable():
    assert d1_misses(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
