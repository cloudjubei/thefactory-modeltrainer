"""§3.6 diagnostic — PRE-REGISTERED claims on what stops steady states at ply 10, from
evidence/c49_s1_budget_probe.json.gz (scripts/c4_steady_budget_probe.py): the eight ply-10 S1-pilot positions that ran
out of the registered budget, re-searched with 8 levels and a 6x budget ("budget") and with 14 levels and the same
budget ("levels"). Found states must pass the solver-free walk and certify."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_budget_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _found() -> dict:
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    assert len(rows) == 16 and all(r.get("verified") and r.get("certified") for r in rows if r["status"] == "found")
    return {s: sum(1 for r in rows if r["setting"] == s and r["status"] == "found") for s in ("budget", "levels")}


def test_c49_s1_probe_a_6x_budget_finds_steady_states_for_3_of_8_ply_10_positions():
    assert _found()["budget"] >= 3


def test_c49_s1_probe_the_budget_reading_is_undecidable():
    assert 1 <= _found()["budget"] <= 2


def test_c49_s1_probe_14_levels_find_at_least_2_more_than_8_levels():
    assert _found()["levels"] - _found()["budget"] >= 2


def test_c49_s1_probe_the_levels_reading_is_undecidable():
    assert _found()["levels"] - _found()["budget"] == 1
