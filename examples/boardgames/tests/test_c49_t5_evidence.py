"""§3.6 T5 — PRE-REGISTERED claims on T4's game written again with each exception flag coded in context too, judged by
harness.floor_t5 from evidence/c49_t5.json.gz (scripts/c4_exception_context.py): T4's game written by
harness.exception_context makes exactly the file's bytes; the file reads back from its bytes alone to that game; every
line of what it reads wins; and it takes 99/100 of T4's 112,644 bits or less."""
from __future__ import annotations

import base64
from functools import lru_cache

import pytest

from games.connect4 import COLS, ROWS, Connect4
from harness.cell_context import decode_strategy as decode_source
from harness.exception_context import decode_strategy, encode_strategy
from harness.floor_t1 import mirror_cells, t1_differences
from harness.floor_t5 import SPEC, t5_report
from harness.steady_state import Facts
from harness.strategy_file import leaf_states
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

FILES = (SPEC["file"], SPEC["source"])
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _facts(game) -> Facts:
    return Facts(game, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})


@lru_cache(maxsize=1)
def _report() -> dict:
    game = Connect4()
    root, cells, levels = game.initial_state(), mirror_cells(ROWS, COLS), SPEC["rule"]["n_levels"]
    e, source = _evidence(SPEC["file"]), _evidence(SPEC["source"])
    facts = _facts(game)
    t4 = decode_source(facts, root, levels, cells, base64.b64decode(source["file"]))
    data = base64.b64decode(e["file"])
    again, bits, _ = encode_strategy(facts, root, levels, cells, t4)
    read = decode_strategy(_facts(game), root, levels, cells, data)
    return t5_report(e, {"bits": bits if again == data else None, "differences": t1_differences(t4, read),
                         "leaves": len(leaf_states(game, root, read["nodes"])["leaves"])}, source, SPEC)


def test_c49_t5_the_file_reads_back_to_t4_s_game_and_every_line_wins():
    assert _report()["decoded"] == "supported"


def test_c49_t5_the_reading_is_undecidable():
    assert _report()["decoded"] == "not_run"


def test_c49_t5_the_file_takes_99_100_of_112_644_bits_or_less():
    assert _report()["size"] == "supported"


def test_c49_t5_the_size_is_undecidable():
    assert _report()["size"] in {"inconclusive", "not_run"}
