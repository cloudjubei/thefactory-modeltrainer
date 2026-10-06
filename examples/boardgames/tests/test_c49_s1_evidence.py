"""§3.6 S1 — PRE-REGISTERED claims on discovered steady states at the frontier of the canonical exact Connect-4
table, judged by harness.floor_s1.s1_report from evidence/c49_s1.json.gz (scripts/c4_steady_states.py over the
seeded sample of evidence/c49_frontier_positions.json.gz and its derived ply-14 frontier): at ply 14 our own SAT
search finds a verified steady state
for at least 30% of the non-trivial sampled positions; and the median found state replaces at least 10x its bits in
3-bit table entries. Each proof passes only on SUPPORTED; the undecidable proofs pass on INCONCLUSIVE or NOT_RUN. The
register pins this file and harness/floor_s1.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_s1 import SPEC, s1_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return s1_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_s1_our_search_finds_a_steady_state_for_30_percent_of_ply_14_frontier_positions():
    assert _report()["coverage"]["verdict"] == "supported"


def test_c49_s1_coverage_is_undecidable():
    assert _report()["coverage"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_s1_a_found_steady_state_replaces_at_least_10_times_its_bits_of_table():
    assert _report()["compression"]["verdict"] == "supported"


def test_c49_s1_compression_is_undecidable():
    assert _report()["compression"]["verdict"] in {"inconclusive", "not_run"}
