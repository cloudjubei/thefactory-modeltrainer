"""§3.6 O1 — finding registered AFTER the data, from evidence/c49_o1.json.gz against G3 (evidence/c49_g3.json.gz, its
opening's moves from books/c4_labels.json.gz) and the scope (evidence/c49_o1_scope.json.gz): the builder's move at
ply 6 shrinks nearly every subtree it differs on. Of the 31, 28 are smaller than G3's and 3 larger (#3, #43, #92);
5 became a single leaf at ply 6 (#13, #21, #52, #54, #127) and 24 played the builder's pick from the scope. The 31
take 20,704 bits against G3's 51,312 (0.40); 27.4 hours of builds."""
from __future__ import annotations

from pathlib import Path

import pytest

from games.connect4 import C4State, Connect4
from harness.evidence import load_evidence
from harness.floor_e1 import e1_control
from harness.floor_g1 import _coded
from harness.floor_o1 import SPEC
from harness.opening import frontier
from harness.steady_state import Facts

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILES = ("c49_o1.json.gz", "c49_g3.json.gz", SPEC["scope"])
LABELS = ROOT / "books" / "c4_labels.json.gz"
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_o1_the_builder_s_move_at_ply_6_shrinks_nearly_every_subtree():
    game = Connect4()
    o1, g3, scope = (load_evidence(EVIDENCE / f) for f in FILES)
    picks = {p["index"]: p["library_pick"] for p in scope["positions"]}
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(LABELS)["positions"]}
    moves = frontier(Facts(game), game.initial_state(), lambda states: [book[game.state_key(s)] for s in states], 8,
                     SPEC["builder"]["n_levels"], SPEC["builder"]["cap"])["moves"]
    rows = {}
    for r in o1["roots"]:
        key = (tuple(r["board"]), r["to_move"])
        root = next(n for b, m, n in r["nodes"] if (tuple(b), m) == key)
        state = C4State(key[0], key[1], None, False)
        rows[r["index"]] = (root.get("move"), _coded(r), e1_control([state], g3, moves)[0])
    assert sorted(rows) == SPEC["indices"]
    assert sorted(i for i, (move, _, _) in rows.items() if move is None) == [13, 21, 52, 54, 127]
    assert sum(move == picks[i] for i, (move, _, _) in rows.items()) == 24
    assert sorted(i for i, (_, arm, g) in rows.items() if arm > g) == [3, 43, 92]
    assert sum(arm < g for _, arm, g in rows.values()) == 28
    assert round(sum(r["build_seconds"] for r in o1["roots"]) / 3600, 1) == 27.4
