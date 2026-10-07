"""§3.6 S5 pilot — finding registered AFTER the data, from evidence/c49_s5_pilot.json.gz (the two roots after S3's
sample, 1 h each, leaves from ply 10 kept with exceptions at >= 10x, exceptions charged in walk order): one root
completes in 15 minutes but only just over 10x — the leaves kept with exceptions are big (the largest over 100K own
positions each) and, together, cover at least as many positions as the whole strategy reaches, so leaves that overlap
pay for the same positions more than once."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s5_pilot.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_s5_pilot_ply_10_leaves_at_10x_give_a_strategy_just_over_10x_from_big_overlapping_leaves():
    e = load_evidence(EVIDENCE / FILE)
    assert e["builder"]["min_leaf_depth"] == 2 and e["builder"]["accept"] == 10 and len(e["roots"]) == 2
    done = [r for r in e["roots"] if r["complete"]]
    assert len(done) == 1 and done[0]["checked"] and done[0]["moves_win"] and done[0]["leaves_certified"]
    r = done[0]
    assert 10 <= 3 * r["own_positions"] / r["bits"]["nodes"] < 12
    excepted = [n for _b, _t, n in r["nodes"] if n.get("exceptions")]
    assert max(n["own"] for n in excepted) > 100_000
    assert sum(n["own"] for n in excepted) >= r["own_positions"]
