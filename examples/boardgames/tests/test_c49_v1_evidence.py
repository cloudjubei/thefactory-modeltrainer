"""§3.6 V1 — PRE-REGISTERED claims on the first batch of every winning move: at the 22 ply-6 positions with more than
one winning move under which G3 pays at least 1,500 bits (evidence/c49_o1_scope.json.gz), the 63 winning moves no
root of E1, O1 or E3 played, each built from ply 6 with that move at the root and otherwise as E1 built its 8, judged
by harness.floor_v1 from evidence/c49_v1.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_v1) against the
whole game J1 assembles (harness.floor_j1) from G3 (evidence/c49_g3.json.gz, its library from c49_g2 and c49_f2) and
the builds of E1, O1 and E3, the ply-6 opening and the ply-8 opening's moves from books/c4_labels.json.gz: at least
58 of the 63 complete within 6 hours, and with them added the whole game takes no more than 97/100 of its 203,774."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from games.connect4 import Connect4
from harness.evidence import load_evidence
from harness.floor_e1 import SPEC as E1
from harness.floor_e3 import SPEC as E3
from harness.floor_g3 import SPEC as G3, g3_library
from harness.floor_j1 import j1_piece
from harness.floor_o1 import SPEC as O1
from harness.floor_v1 import SPEC, v1_report, v1_size
from harness.opening import frontier
from harness.steady_state import Facts, map_bits

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_v1.json.gz"
EARLIER = dict(zip(SPEC["earlier"], (E1, O1, E3)))
FILES = (FILE, *EARLIER, SPEC["scope"], "c49_g3.json.gz", "c49_g2.json.gz", "c49_f2.json.gz")
LABELS = ROOT / "books" / "c4_labels.json.gz"
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=8)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _assembly() -> dict:
    game = Connect4()
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(LABELS)["positions"]}

    def winning(states):
        return [book[game.state_key(s)] for s in states]

    b = G3["builder"]
    shallow = frontier(Facts(game), game.initial_state(), winning, 6, b["n_levels"], b["cap"])
    moves = frontier(Facts(game), game.initial_state(), winning, 8, b["n_levels"], b["cap"])["moves"]
    library = g3_library(_evidence("c49_g2.json.gz"), _evidence("c49_f2.json.gz"))
    below = {}
    for s in shallow["frontier"]:
        child = game.step(s, moves[game.state_key(s)])
        below[game.state_key(s)] = [game.state_key(c) for c in (game.step(child, a) for a in game.legal_actions(child))
                                    if not game.is_terminal(c)]
    base = {(tuple(r["board"]), r["to_move"]): j1_piece(r, library) for r in _evidence("c49_g3.json.gz")["roots"]}
    return {"opening": {"moves": len(shallow["moves"]), "trivial": len(shallow["trivial"])}, "below": below,
            "base": base,
            "map_cost": {i: map_bits({int(c): k for c, k in m.items()}, e, b["level_bits"])
                         for i, (m, e) in enumerate(library, 1)},
            "library": library}


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return v1_size(_evidence(), earlier, _evidence(SPEC["scope"]), _assembly(), SPEC)


def test_c49_v1_58_of_63_unbuilt_winning_moves_complete_within_6_hours():
    assert v1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_v1_completion_is_undecidable():
    assert v1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_v1_the_whole_game_with_the_batch_takes_97_100_of_j1_s_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_v1_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
