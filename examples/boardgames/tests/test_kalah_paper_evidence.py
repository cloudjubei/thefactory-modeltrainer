"""Kalah rung — games/kalah.py against Irving, Donkers & Uiterwijk (2000). Two claims, judged by
harness.kalah_paper.paper_report from evidence/kalah_paper_check.json.gz (scripts/kalah_paper_check.py): the exact
solver's value from the start agrees with the paper's Table 10 (win/draw/loss) and Table 9 (margins) on every
registered shape; and the paper's perfect games replay to their stated margins under its stated rules better than under
any other reading, the 4(6) line breaking where registered with no one-character repair. Each proof passes only on
SUPPORTED; the undecidable proof passes on NOT_RUN. The register pins this file and harness/kalah_paper.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.kalah_paper import SPEC, paper_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "kalah_paper_check.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return paper_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_kalah_the_solver_agrees_with_the_paper_s_values_on_every_registered_shape():
    assert _report()["values"]["verdict"] == "supported"


def test_kalah_the_paper_s_perfect_games_fit_its_stated_rules_best_and_only_4_6_breaks():
    assert _report()["replays"]["verdict"] == "supported"


def test_kalah_paper_check_is_undecidable():
    assert _report()["integrity"]
