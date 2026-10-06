"""§3.6 S2 pilot — registered AFTER the data, from evidence/c49_s2_pilot.json.gz (two ply-10 roots after the registered
eight, a 20-minute cap instead of 2 hours): one root got a complete certified strategy — 2,791 bits for 31,701
first-player positions (34x under a 3-bit table) — the other ran out of time after 180 nodes."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s2_pilot.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_s2_pilot_one_complete_certified_strategy_34_times_under_its_table():
    roots = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    done = roots[2556]
    assert done["complete"] and done["checked"] and done["moves_win"] and done["leaves_certified"]
    assert done["bits"]["nodes"] == 2791 and done["own_positions"] == 31701
    assert 3 * done["own_positions"] / done["bits"]["nodes"] > 34
    assert not roots[1149]["complete"] and len(roots[1149]["nodes"]) == 180
