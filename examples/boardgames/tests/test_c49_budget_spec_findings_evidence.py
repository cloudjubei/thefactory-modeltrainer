"""§3.6 — finding registered AFTER the data: P2's, W2's, W3's and W4's builders were registered with budgets [[0, 1200]]
and described as "20 minutes at the root, 30 s below", but a budget entry applies to its depth and every depth below
it (harness.strategy_builder), so every search had 20 minutes. Every build with 10 or more searches averaged over
500 s a search — at the described budgets such a build would average under 150 s."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_p2.json.gz", "c49_w2.json.gz", "c49_w3.json.gz", "c49_w4.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_every_search_had_the_root_s_twenty_minutes():
    long_builds = 0
    for name in FILES:
        e = load_evidence(EVIDENCE / name)
        assert e["builder"]["budgets"] == [[0, 1200.0]]
        for r in e["roots"]:
            tried = r["searches"]["tried"]
            if tried >= 10:
                long_builds += 1
                assert r["build_seconds"] / tried > 500
                assert (1200 + 30 * (tried - 1)) / tried < 150
    assert long_builds >= 6
