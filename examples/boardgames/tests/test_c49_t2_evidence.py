"""§3.6 T2 — PRE-REGISTERED claims on T1's file with its maps shrunk to what its leaves play, judged by
harness.floor_t2 from evidence/c49_t2.json.gz (scripts/c4_map_orders.py): the maps its orders plan, written into T1's
game, make exactly the file's bytes; the file reads back from its bytes alone to that game; every line of what it reads
wins; and it takes 19/20 of T1's 130,005 bits or less."""
from __future__ import annotations

import base64
from functools import lru_cache

import pytest

from games.connect4 import COLS, ROWS, C4State, Connect4
from harness.floor_t1 import mirror_cells, t1_differences
from harness.floor_t2 import SPEC, t2_orders, t2_report
from harness.map_orders import merged, shrink
from harness.steady_state import Facts
from harness.strategy_file import decode_strategy, encode_strategy, leaf_states
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

FILES = (SPEC["file"], SPEC["source"])
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _facts(game) -> Facts:
    return Facts(game, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})


def _state(key) -> C4State:
    board, to_move = key
    return C4State(tuple(board), to_move, None, False)


@lru_cache(maxsize=1)
def _report() -> dict:
    game = Connect4()
    root, cells, levels = game.initial_state(), mirror_cells(ROWS, COLS), SPEC["rule"]["n_levels"]
    e, source = _evidence(SPEC["file"]), _evidence(SPEC["source"])
    facts = _facts(game)
    t1 = decode_strategy(facts, root, levels, cells, base64.b64decode(source["file"]))
    orders = t2_orders((int(i), s, w) for i, (s, w) in e["orders"].items())
    shrunk = shrink(t1, merged(orders, levels), {}, facts, levels, _state)
    data = base64.b64decode(e["file"])
    again, bits, _ = encode_strategy(facts, root, levels, cells, shrunk)
    read = decode_strategy(_facts(game), root, levels, cells, data)
    return t2_report(e, {"bits": bits if again == data else None, "differences": t1_differences(shrunk, read),
                         "leaves": len(leaf_states(game, root, read["nodes"])["leaves"])}, source, SPEC)


def test_c49_t2_the_file_reads_back_to_the_shrunk_game_and_every_line_wins():
    assert _report()["decoded"] == "supported"


def test_c49_t2_the_reading_is_undecidable():
    assert _report()["decoded"] == "not_run"


def test_c49_t2_the_file_takes_19_20_of_130_005_bits_or_less():
    assert _report()["size"] == "supported"


def test_c49_t2_the_size_is_undecidable():
    assert _report()["size"] in {"inconclusive", "not_run"}
