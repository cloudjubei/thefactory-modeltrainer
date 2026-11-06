"""§3.6 T1 — PRE-REGISTERED claims on Z2's game written as one file, judged by harness.floor_t1 from
evidence/c49_t1.json.gz (scripts/c4_strategy_file.py): the file reads back from its bytes alone to the game that was
written, every line of what it reads wins, and it takes 9/10 of the 162,752 bits Z2's game was counted at or less."""
from __future__ import annotations

import base64
from functools import lru_cache

import pytest

from games.connect4 import COLS, ROWS, Connect4
from harness.evidence import load_evidence
from harness.floor_g3 import SPEC as G3
from harness.floor_g4 import SPEC as G4
from harness.floor_t1 import (SPEC, mirror_cells, t1_differences, t1_game, t1_opening, t1_pieces, t1_report,
                              t1_rule)
from harness.floor_u2 import SPEC as U2
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC as X4
from harness.floor_z1 import SPEC as Z1
from harness.floor_z2 import SPEC as Z2, _runs, _taken, z2_size
from harness.steady_state import Facts
from harness.strategy_file import decode_strategy, encode_strategy, leaf_states
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER
from tests.test_c49_z2_evidence import EVIDENCE, FILES as Z2_FILES, LABELS, _evidence

FILES = (SPEC["file"], *Z2_FILES)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _opening() -> dict:
    return t1_opening(Connect4(), load_evidence(LABELS)["positions"], G3["builder"], SPEC["plies"])


def _mirrors() -> dict:
    specs = {"g3": G3, "g4": G4, "z": Z1, "x2": X2, "x3": X3, "x4": X4, "z2": Z2}
    assert t1_rule([s["builder"] for s in (*specs.values(), *EARLIER.values())]) == SPEC["rule"]
    assert not any(s["builder"].get("mirror") for s in EARLIER.values())
    return {"earlier": False, **{name: bool(s["builder"].get("mirror")) for name, s in specs.items()}}


def _z2_pieces() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    a, g3, g4 = _assembly(), _evidence("c49_g3.json.gz"), _evidence("c49_g4.json.gz")
    z, x2 = _evidence("c49_z1.json.gz"), _evidence("c49_x2.json.gz")
    x3, x4 = _evidence("c49_x3.json.gz"), _evidence("c49_x4.json.gz")
    state = z2_size(_evidence(), (x4, X4), (x3, X3), (x2, X2), (z, Z1), earlier, (g4, G4), g3, a, U2, Z2)["state"]
    rows = [r for run, _ in earlier for r in run["roots"]]
    runs = {**_runs(rows, g3, g4, z, x2, x3, x4), "z2": {r["index"]: r for r in _evidence()["roots"]}}
    return t1_pieces(lambda: _taken(state, runs, a["below"]), _mirrors())


def _g3_pieces() -> dict:
    a = _assembly()
    kids = {c for children in a["below"].values() for c in children}
    rows = [r for r in _evidence("c49_g3.json.gz")["roots"] if (tuple(r["board"]), r["to_move"]) in kids]
    return t1_pieces(lambda: [("g3", r["index"], r) for r in rows], _mirrors())


@lru_cache(maxsize=1)
def _report() -> dict:
    game = Connect4()
    root, cells, levels = game.initial_state(), mirror_cells(ROWS, COLS), SPEC["rule"]["n_levels"]
    facts = Facts(game, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})
    e = _evidence(SPEC["file"])
    data = base64.b64decode(e["file"])
    read = decode_strategy(facts, root, levels, cells, data)
    source = t1_game(facts, root, cells, _opening(), _z2_pieces())
    again, bits, _ = encode_strategy(facts, root, levels, cells, source)
    return t1_report(e, {"bits": bits if again == data else None, "differences": t1_differences(source, read),
                         "leaves": len(leaf_states(game, root, read["nodes"])["leaves"])}, SPEC)


def test_c49_t1_the_file_reads_back_to_the_game_and_every_line_wins():
    assert _report()["decoded"] == "supported"


def test_c49_t1_the_reading_is_undecidable():
    assert _report()["decoded"] == "not_run"


def test_c49_t1_the_file_takes_9_10_of_162_752_bits_or_less():
    assert _report()["size"] == "supported"


def test_c49_t1_the_size_is_undecidable():
    assert _report()["size"] in {"inconclusive", "not_run"}
