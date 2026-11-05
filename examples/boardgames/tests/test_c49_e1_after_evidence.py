"""§3.6 E1 — finding registered AFTER the data, from evidence/c49_e1.json.gz against G3 (evidence/c49_g3.json.gz, its
opening's moves from books/c4_labels.json.gz): the saving is the move at ply 6, not an earlier leaf. No root is a
leaf. On #115, #130 and #145 the builder played another winning move than the opening's lowest column, and the three
subtrees fell from G3's 18,156 bits to 2,840; on the five where it kept the opening's move they take 30,235 against
28,882. 16.9 hours of builds."""
from __future__ import annotations

from pathlib import Path

import pytest

from games.connect4 import C4State, Connect4
from harness.evidence import load_evidence
from harness.floor_e1 import SPEC, e1_control
from harness.floor_g1 import _coded
from harness.opening import frontier
from harness.steady_state import Facts

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILES = ("c49_e1.json.gz", "c49_g3.json.gz")
LABELS = ROOT / "books" / "c4_labels.json.gz"
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_e1_the_saving_is_the_builder_s_move_at_ply_6():
    game = Connect4()
    e1, g3 = (load_evidence(EVIDENCE / f) for f in FILES)
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(LABELS)["positions"]}
    moves = frontier(Facts(game), game.initial_state(), lambda states: [book[game.state_key(s)] for s in states], 8,
                     SPEC["builder"]["n_levels"], SPEC["builder"]["cap"])["moves"]
    rows = {}
    for r in e1["roots"]:
        key = (tuple(r["board"]), r["to_move"])
        root = next(n for b, m, n in r["nodes"] if (tuple(b), m) == key)
        assert "leaf" not in root and root["move"] in book[key]
        state = C4State(key[0], key[1], None, False)
        rows[r["index"]] = (root["move"] != moves[key], _coded(r), e1_control([state], g3, moves)[0])
    assert sorted(i for i, (moved, _, _) in rows.items() if moved) == [115, 130, 145]
    for moved, arm, control in ((True, 2_840, 18_156), (False, 30_235, 28_882)):
        picked = [v for v in rows.values() if v[0] == moved]
        assert (sum(v[1] for v in picked), sum(v[2] for v in picked)) == (arm, control)
    assert round(sum(r["build_seconds"] for r in e1["roots"]) / 3600, 1) == 16.9
