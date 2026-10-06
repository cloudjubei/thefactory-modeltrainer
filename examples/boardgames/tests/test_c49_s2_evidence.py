"""§3.6 S2 — PRE-REGISTERED claims on complete certified first-player strategies (table moves + steady-state leaves)
for the rest of the game from 8 sampled ply-10 positions of the canonical exact Connect-4 table, judged by
harness.floor_s2.s2_report from evidence/c49_s2.json.gz (scripts/c4_strategy_s2.py): at least 6 of the 8 complete
within the 2-hour cap; and over completed roots the median strategy is at least 10x smaller than the same strategy as a
3-bit-per-position table. Each proof passes only on SUPPORTED; the undecidable proofs pass on INCONCLUSIVE or NOT_RUN.
The register pins this file and harness/floor_s2.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_s2 import SPEC, s2_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s2.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return s2_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_s2_six_of_eight_ply_10_roots_get_a_complete_certified_strategy_within_the_cap():
    assert _report()["complete"]["verdict"] == "supported"


def test_c49_s2_completion_is_undecidable():
    assert _report()["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_s2_the_strategies_are_at_least_10_times_smaller_than_their_table():
    assert _report()["compression"]["verdict"] == "supported"


def test_c49_s2_compression_is_undecidable():
    assert _report()["compression"]["verdict"] in {"inconclusive", "not_run"}
