"""Plan §2.3 E2 — PRE-REGISTERED claims on how small certified first-player play in Kalah is, judged by
harness.floor_e2.e2_report from evidence/kalah_strategy_size.json.gz (scripts/kalah_strategy_size.py): on every shape
whose game graph holds at least 1,000 positions the tree-minimising certified strategy decides at no more than 5% of
them; and on at least two thirds of those shapes it needs at most half the canonical strategy's decisions. Each proof
passes only on SUPPORTED; the undecidable proofs pass on INCONCLUSIVE or NOT_RUN. The register pins this file and
harness/floor_e2.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_e2 import SPEC, e2_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "kalah_strategy_size.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return e2_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_kalah_a_certified_first_player_strategy_decides_at_no_more_than_5_percent_of_the_game():
    assert _report()["fraction"]["verdict"] == "supported"


def test_kalah_the_fraction_is_undecidable():
    assert _report()["fraction"]["verdict"] in {"inconclusive", "not_run"}


def test_kalah_choosing_value_keeping_moves_halves_the_strategy_on_two_thirds_of_shapes():
    assert _report()["halving"]["verdict"] == "supported"


def test_kalah_the_halving_is_undecidable():
    assert _report()["halving"]["verdict"] in {"inconclusive", "not_run"}
