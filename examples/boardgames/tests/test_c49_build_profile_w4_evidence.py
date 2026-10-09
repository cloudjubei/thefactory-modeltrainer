"""§3.6 — finding registered AFTER the data, from two profiled W4-style builds of #230 with the C walk
(evidence/c49_build_profile_w4.json.gz: the root's long search; evidence/c49_build_profile_w4_below_root.json.gz: the
root's search cut to 60 s): with the C walk a search makes tens of times more evaluations than with Python, the
searches keep to their budgets and take most of the build, and inside the C walk's wrapper building the exceptions'
Python keys took over a third of the walk's time."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_build_profile_w4.json.gz", "c49_build_profile_w4_below_root.json.gz", "c49_build_profile.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _cum(e, name):
    return next(t["seconds"] for t in e["top_cumulative"] if t["function"].endswith(name))


def test_c49_the_c_walk_multiplies_evaluations_and_its_key_building_was_a_third_of_it():
    root, below, python = (load_evidence(EVIDENCE / f) for f in FILES)
    assert root["searches"][0]["evaluations"] > 20 * python["searches"][0]["evaluations"]
    for e in (root, below):
        assert e["search_seconds"] > 0.85 * e["wall_seconds"]
        assert all(s["seconds"] <= s["budget"] + 1 for s in e["searches"])
    assert _cum(below, "native_leaf.py:55:_board") > _cum(below, "native_leaf.py:62:needed") / 3
