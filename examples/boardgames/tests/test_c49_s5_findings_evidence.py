"""§3.6 S5 — finding registered AFTER the data, from evidence/c49_s5.json.gz beside S3 (evidence/c49_s3.json.gz) and
S4 re-costed in walk order (evidence/c49_s4_exception_coding.json.gz): on #3591, the root all three completed, S5's
strategy — leaves from ply 10, kept with exceptions at >= 30x — is the smallest, more than 20% under both."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_s5.json.gz", "c49_s3.json.gz", "c49_s4_exception_coding.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _root(name, index=3591):
    return {r["index"]: r for r in load_evidence(EVIDENCE / name)["roots"]}[index]


def test_c49_s5_ply_10_leaves_give_the_smallest_3591_strategy_by_over_20_percent():
    s5, s3, s4 = _root(FILES[0]), _root(FILES[1]), _root(FILES[2])
    assert s5["complete"] and s5["checked"] and s5["moves_win"] and s5["leaves_certified"]
    assert s3["complete"] and s4["complete"]
    assert s5["bits"]["nodes"] < 0.8 * s3["bits"]["nodes"] and s5["bits"]["nodes"] < 0.8 * s4["nodes_walk"]
