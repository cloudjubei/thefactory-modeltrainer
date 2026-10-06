"""§3.6 — finding registered AFTER the data, from evidence/c49_s1_leaf_probe.json.gz beside the earlier searches of
the same eight ply-10 positions: local search scored by leaf size ends at a pure steady state (no exceptions) for 4
of 8 within 30 minutes, where SAT with a 6x budget found 1 (h162) and the failure-count local search 0 (h166) — the
score, not the language, was the ply-10 limit."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_s1_leaf_probe.json.gz", "c49_s1_local_probe.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_s1_size_scored_search_finds_4_pure_steady_states_where_failure_counting_found_none():
    leaf = load_evidence(EVIDENCE / FILES[0])["rows"]
    local = load_evidence(EVIDENCE / FILES[1])["rows"]
    assert {r["index"] for r in leaf} == {r["index"] for r in local} and len(leaf) == 8
    pure = [r for r in leaf if r["status"] == "found"]
    assert len(pure) == 4 and all(not r["exceptions"] and r["verified"] and r["certified"] for r in pure)
    assert not any(r["status"] == "found" for r in local)
