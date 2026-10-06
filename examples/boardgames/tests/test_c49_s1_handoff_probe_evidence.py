"""§3.6 — PRE-REGISTERED claims on the SAT hand-off, from evidence/c49_s1_handoff_probe.json.gz
(scripts/c4_steady_handoff_probe.py): on the 4 ply-10 positions where size-scored local search ended with exceptions,
SAT started at local search's best map with its exception positions constrained first ("hinted") finds a pure,
verified, certified steady state for at least 2 within 30 minutes and 200K constraints, and for more positions than
the same SAT search without them ("cold")."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_handoff_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _found() -> dict:
    data = load_evidence(EVIDENCE / FILE)
    assert data["settings"]["budget"]["max_constraints"] == 200_000 and data["settings"]["budget"]["seconds"] == 1800.0
    rows = data["rows"]
    assert len(rows) == 8 and {r["arm"] for r in rows} == {"hinted", "cold"}
    done = [r for r in rows if r["status"] == "found"]
    assert all(r["verified"] and r["certified"] for r in done)
    return {arm: sum(r["arm"] == arm for r in done) for arm in ("hinted", "cold")}


def test_c49_s1_hinted_sat_finds_2_of_4_near_miss_positions():
    assert _found()["hinted"] >= 2


def test_c49_s1_hinted_sat_finds_one_near_miss_position():
    assert _found()["hinted"] == 1


def test_c49_s1_hinted_sat_finds_more_than_cold_sat():
    assert _found()["hinted"] > _found()["cold"]
