"""§3.6 — finding registered AFTER the data, from two profiled builds of W1's frontier position #316 under P2's
settings (evidence/c49_build_profile.json.gz: the 20-minute root search, capped at 15 minutes;
evidence/c49_build_profile_below_root.json.gz: the root's search cut to 60 s): the leaf searches take over 95% of a
build's time, and near the root one step of a search — a full walk of the leaf — takes minutes, so searches there
overrun their budgets by more than half again: the budget is checked only between steps."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_build_profile.json.gz", "c49_build_profile_below_root.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_leaf_searches_take_the_build_and_overrun_their_budgets_near_the_root():
    for name in FILES:
        e = load_evidence(EVIDENCE / name)
        assert e["index"] == 316 and e["search_seconds"] > 0.95 * e["wall_seconds"]
    below = load_evidence(EVIDENCE / FILES[1])["searches"]
    near = [s for s in below if s["pieces"] <= 10]
    assert near and all(s["seconds"] > 1.5 * s["budget"] and s["own_positions"] > 100_000 for s in near)
    assert min(s["seconds"] / s["evaluations"] for s in near) > 60
