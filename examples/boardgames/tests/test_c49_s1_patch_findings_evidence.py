"""§3.6 — finding registered AFTER the data, from evidence/c49_s1_patch_probe.json.gz (h168 refuted): the local
search's best map at ply 10 is the empty map on seven of the eight positions, and there its one failing line is the
root itself — the map gives no move there, and the score stops at an undefined position, so every map that does give
a root move exposes more failures below and is rejected. h167's 'one line short' was an artefact of the score."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_patch_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_s1_the_best_local_map_is_empty_and_undefined_at_the_root_on_7_of_8():
    from games.connect4 import C4State, Connect4
    from harness.steady_state import Facts, choose

    rows = load_evidence(EVIDENCE / FILE)["rows"]
    pilot = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_s1_pilot.json.gz")["results"]}
    facts = Facts(Connect4())
    empty = [r for r in rows if not r["levels"] and r["best_failures"] == 1]
    assert len(rows) == 8 and len(empty) == 7
    for r in empty:
        p = pilot[r["index"]]
        assert choose(facts, C4State(tuple(p["board"]), p["to_move"], None, False), {}, 8) is None


def test_c49_s1_no_ply_10_position_got_a_map_plus_exceptions_leaf():
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    assert len(rows) == 8 and all(r["patch"] == "too_many" and "bits" not in r for r in rows)
