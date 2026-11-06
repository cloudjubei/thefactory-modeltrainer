"""§3.6 T1 — finding registered AFTER the data, from evidence/c49_t1.json.gz and the evidence of the runs up to Z2: what
the file holds and what it pays against the count. It takes 130,005 bits (16,251 bytes), 0.7988 of the 162,752 the game
was counted at, and read from its bytes alone it holds 4,052 positions, 680 moves and 3,372 leaves, 616 of them on the
empty map, 43 reading a map mirrored and 129 with 1,625 exceptions, and 1,052 maps of 19,715 cells, 514 of them read
once. Maps take 89,289 bits against the count's 99,613 (70,435 for the library, 29,178 for the pieces' own), the
exceptions 21,051 against 21,815, and the tree 19,663 against the 41,324 left in the count: of the 32,747 bits the file
saves, 21,661 are the tree's. The levels are the largest part, 57,615 bits, 2.92 a cell of the 3 a level can take."""
from __future__ import annotations

import base64

import pytest

from games.connect4 import COLS, ROWS, Connect4
from harness.floor_t1 import SPEC, mirror_cells
from harness.steady_state import Facts
from harness.strategy_file import decode_strategy
from tests.test_c49_t1_evidence import FILES, _z2_pieces
from tests.test_c49_z2_evidence import EVIDENCE, LABELS, _evidence

LIBRARY_BITS = 70_435
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_t1_the_file_takes_130_005_bits_and_two_thirds_of_what_it_saves_is_the_tree():
    e = _evidence(SPEC["file"])
    data = base64.b64decode(e["file"])
    assert (e["bits"], len(data)) == (130_005, 16_251) and round(e["bits"] / SPEC["control_bits"], 4) == 0.7988
    game = Connect4()
    facts = Facts(game, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})
    read = decode_strategy(facts, game.initial_state(), SPEC["rule"]["n_levels"], mirror_cells(ROWS, COLS), data)
    nodes, maps = read["nodes"], read["maps"]
    leaves = [n for n in nodes.values() if "leaf" in n]
    assert (len(nodes), len(nodes) - len(leaves), len(leaves)) == (4_052, 680, 3_372)
    assert sum(not maps[n["leaf"]] for n in leaves) == 616 and sum(bool(n.get("mirrored")) for n in leaves) == 43
    assert sum(bool(n.get("exceptions")) for n in leaves) == 129
    assert sum(len(n.get("exceptions", {})) for n in leaves) == 1_625
    reads = [sum(n["leaf"] == i for n in leaves) for i in range(1, len(maps))]
    assert (len(reads), sum(len(m) for m in maps), reads.count(1)) == (1_052, 19_715, 514)

    parts = e["parts"]
    file_maps = round(parts["cells"] + parts["levels"])
    file_exceptions = round(parts["excepted"] + parts["exception_flags"] + parts["exception_moves"])
    file_tree = round(parts["kind"] + parts["move"] + parts["reference"] + parts["mirrored"])
    assert (file_maps, file_exceptions, file_tree) == (89_289, 21_051, 19_663)
    rows = [r for r, _ in _z2_pieces().values()]
    own = sum(r["bits"]["map_bits"] for r in rows)
    exceptions = sum(r["bits"]["exception_bits"] for r in rows)
    tree = SPEC["control_bits"] - LIBRARY_BITS - own - exceptions
    assert (own, exceptions, tree) == (29_178, 21_815, 41_324)
    assert SPEC["control_bits"] - e["bits"] == 32_747 and tree - file_tree == 21_661
    assert round(parts["levels"]) == 57_615 and round(parts["levels"] / 19_715, 2) == 2.92
