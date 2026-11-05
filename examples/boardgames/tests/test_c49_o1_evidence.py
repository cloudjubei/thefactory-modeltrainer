"""§3.6 O1 — PRE-REGISTERED claims on the 31 ply-6 positions beyond E1's 8 where the builder's move rule, given G3's
library, picks another move than the opening (evidence/c49_o1_scope.json.gz), each rebuilt from ply 6 as E1 built its
8, judged by harness.floor_o1 from evidence/c49_o1.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_o1)
against G3 (evidence/c49_g3.json.gz, its library from c49_g2 and c49_f2, its opening's moves from
books/c4_labels.json.gz): at least 29 of the 31 complete within 6 hours, and they take no more than three quarters of
G3's 51,312 bits for the same subtrees."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from games.connect4 import Connect4
from harness.evidence import load_evidence
from harness.floor_g3 import g3_library
from harness.floor_o1 import SPEC, o1_report, o1_size
from harness.opening import frontier
from harness.steady_state import Facts

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_o1.json.gz"
FILES = (FILE, SPEC["scope"], "c49_g3.json.gz", "c49_g2.json.gz", "c49_f2.json.gz")
LABELS = ROOT / "books" / "c4_labels.json.gz"
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=5)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    game = Connect4()
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(LABELS)["positions"]}
    moves = frontier(Facts(game), game.initial_state(), lambda states: [book[game.state_key(s)] for s in states], 8,
                     SPEC["builder"]["n_levels"], SPEC["builder"]["cap"])["moves"]
    library = g3_library(_evidence("c49_g2.json.gz"), _evidence("c49_f2.json.gz"))
    return o1_size(_evidence(), _evidence("c49_g3.json.gz"), library, moves, _evidence(SPEC["scope"]), SPEC)


def test_c49_o1_29_of_31_ply_6_subtrees_complete_within_6_hours():
    assert o1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_o1_completion_is_undecidable():
    assert o1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_o1_the_builder_s_move_at_ply_6_makes_its_subtrees_a_quarter_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_o1_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
