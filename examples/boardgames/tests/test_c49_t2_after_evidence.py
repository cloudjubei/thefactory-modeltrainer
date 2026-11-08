"""§3.6 T2 — finding registered AFTER the data, from evidence/c49_t2.json.gz and T1's: what shrinking the maps saved.
The file takes 118,197 bits (14,775 bytes), 0.9092 of T1's 130,005, and read from its bytes alone it holds the same
4,052 positions, 680 moves and 3,372 leaves, 616 of them on the empty map, 43 reading a map mirrored and 129 with
1,625 exceptions, none of them dropped, and 1,008 maps of 18,444 cells, 468 of them read once. Maps take 77,045 bits
against T1's 89,289, the exceptions 21,054 against 21,051 and the tree 20,099 against 19,663: of the 11,808 bits
saved, the levels give 11,519, at 2.50 bits a cell against 2.92, for the lowest levels crowd the bottom (5,432 cells
on level 0, 73 on level 7)."""
from __future__ import annotations

import base64
from collections import Counter

import pytest

from games.connect4 import COLS, ROWS, Connect4
from harness.floor_t1 import mirror_cells
from harness.floor_t2 import SPEC
from harness.steady_state import Facts
from harness.strategy_file import decode_strategy
from tests.test_c49_t2_evidence import FILES
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _whole(parts: dict) -> tuple:
    return (round(parts["cells"] + parts["levels"]),
            round(parts["excepted"] + parts["exception_flags"] + parts["exception_moves"]),
            round(parts["kind"] + parts["move"] + parts["reference"] + parts["mirrored"]))


def test_c49_t2_the_file_takes_118_197_bits_and_the_levels_give_what_it_saves():
    e, t1 = _evidence(SPEC["file"]), _evidence(SPEC["source"])
    data = base64.b64decode(e["file"])
    assert (e["bits"], len(data)) == (118_197, 14_775) and round(e["bits"] / SPEC["control_bits"], 4) == 0.9092
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
    assert (len(reads), sum(len(m) for m in maps), reads.count(1)) == (1_008, 18_444, 468)
    levels = Counter(v for m in maps for v in m.values())
    assert (levels[0], levels[7]) == (5_432, 73) and all(levels[k] > levels[k + 1] for k in range(7))

    assert _whole(e["parts"]) == (77_045, 21_054, 20_099) and _whole(t1["parts"]) == (89_289, 21_051, 19_663)
    assert SPEC["control_bits"] - e["bits"] == 11_808
    assert round(t1["parts"]["levels"]) - round(e["parts"]["levels"]) == 11_519
    assert round(e["parts"]["levels"] / 18_444, 2) == 2.50
