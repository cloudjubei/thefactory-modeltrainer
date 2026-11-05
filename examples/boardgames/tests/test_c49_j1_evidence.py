"""§3.6 J1 — finding registered AFTER the data: the whole game assembled (harness.floor_j1) from G3
(evidence/c49_g3.json.gz, its library from c49_g2 and c49_f2) and the ply-6 builds of E1, O1 and E3
(evidence/c49_e1.json.gz, c49_o1.json.gz, c49_e3.json.gz), the ply-6 opening and the ply-8 opening's moves from
books/c4_labels.json.gz. With no builds the assembly is G3's whole game exactly, 280,640 bits; choosing on the whole
game's bits it takes 203,774 (0.73): 30 of the 39 ply-6 positions with builds take one, 503 of G3's 671 strategies
and 891 of its 1,207 library maps remain, and every piece is complete and passed its independent checks."""
from __future__ import annotations

from pathlib import Path

import pytest

from games.connect4 import Connect4
from harness.evidence import load_evidence
from harness.floor_e3 import _root_move
from harness.floor_g3 import SPEC as G3, g3_library
from harness.floor_j1 import j1_bits, j1_choose, j1_piece, j1_valid
from harness.opening import frontier
from harness.steady_state import Facts, map_bits

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
BUILDS = ("c49_e1.json.gz", "c49_o1.json.gz", "c49_e3.json.gz")
FILES = ("c49_g3.json.gz", "c49_g2.json.gz", "c49_f2.json.gz", *BUILDS)
LABELS = ROOT / "books" / "c4_labels.json.gz"
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_j1_the_assembled_whole_game_takes_0_73_of_g3_s_bits():
    game = Connect4()
    g3, g2, f2 = (load_evidence(EVIDENCE / f) for f in FILES[:3])
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(LABELS)["positions"]}

    def winning(states):
        return [book[game.state_key(s)] for s in states]

    b = G3["builder"]
    shallow = frontier(Facts(game), game.initial_state(), winning, 6, b["n_levels"], b["cap"])
    moves = frontier(Facts(game), game.initial_state(), winning, 8, b["n_levels"], b["cap"])["moves"]
    library = g3_library(g2, f2)
    cost = {i: map_bits({int(c): k for c, k in m.items()}, e, b["level_bits"]) for i, (m, e) in enumerate(library, 1)}
    assert all(j1_valid(r, library) for r in g3["roots"])
    base = {(tuple(r["board"]), r["to_move"]): j1_piece(r, library) for r in g3["roots"]}
    below = {}
    for s in shallow["frontier"]:
        child = game.step(s, moves[game.state_key(s)])
        below[game.state_key(s)] = [game.state_key(c) for c in (game.step(child, a) for a in game.legal_actions(child))
                                    if not game.is_terminal(c)]
    opening = {"moves": len(shallow["moves"]), "trivial": len(shallow["trivial"])}
    assert j1_bits(opening, below, base, {}, cost) == {"bits": 280_640, "kept": 148, "trivial": 149,
                                                       "strategies": 671, "library_maps": 1207}
    candidates = {}
    for name in BUILDS:
        for r in load_evidence(EVIDENCE / name)["roots"]:
            assert j1_valid(r, library)
            candidates.setdefault((tuple(r["board"]), r["to_move"]), []).append(
                (f"{name[4:6]}#{r['index']}:{_root_move(r)}", j1_piece(r, library)))
    assert len(candidates) == 39 and sum(map(len, candidates.values())) == 52
    chosen, bits = j1_choose(opening, below, base, candidates, cost)
    assert bits == 203_774 and len(chosen) == 30
    assert sorted(chosen.values()) == [
        "e1#115:4", "e1#130:5", "e1#145:2", "e3#146:4", "e3#18:4", "e3#69:1", "e3#79:4", "o1#103:3", "o1#105:3",
        "o1#113:2", "o1#118:2", "o1#124:2", "o1#125:2", "o1#126:2", "o1#135:4", "o1#140:4", "o1#14:4", "o1#21:None",
        "o1#28:2", "o1#42:3", "o1#48:3", "o1#52:None", "o1#54:None", "o1#77:2", "o1#78:2", "o1#80:4", "o1#81:1",
        "o1#89:1", "o1#92:3", "o1#97:3"]
    built = {k: dict(candidates[k])[label] for k, label in chosen.items()}
    assert j1_bits(opening, below, base, built, cost) == {"bits": 203_774, "kept": 118, "trivial": 144,
                                                          "strategies": 503, "library_maps": 891}
