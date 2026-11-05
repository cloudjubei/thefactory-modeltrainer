"""§3.6 G4 — finding registered AFTER the data, from evidence/c49_g4.json.gz with the whole game assembled as
harness.floor_g4 assembles it: all 409 rebuilds complete (55.2 hours against G3's 67.0 for the same positions, each
complete within 2,594 seconds and stopped within 5,171, under the 2-hour cap), and the whole game takes 184,556 bits —
0.958 of 192,591. Alone, 86 rebuilds are smaller than G3's strategy and 204 play a map beyond G3's library; the 409
together take 55,928 bits against G3's 61,427, but put in all at once they grow the game to 195,394, for the new maps
they pay. Sending back one at a time the rebuild whose return shrinks the game most, 333 go back and 76 stay (22,674
bits against G3's 29,270), 184,565 bits; the ply-6 builds chosen again keep the same 51 positions and change one
build, #44's move 1 (344 bits) for its move 3 (433), 184,556. The parts: opening 845, library 71,109 (816 maps, 83
of them beyond G3's library), 409 strategies 54,831, builds at 51 ply-6 positions 57,771."""
from __future__ import annotations

import pytest

from harness.floor_g4 import SPEC, _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece
from harness.floor_v2 import v2_assemble
from harness.steady_state import map_bits
from tests.test_c49_g4_evidence import EARLIER, EVIDENCE, FILES, LABELS, _evidence
from tests.test_c49_v1_evidence import _assembly

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_g4_saves_4_2_percent_from_76_rebuilds_kept_of_409():
    g4, g3 = _evidence()["roots"], {r["index"]: r for r in _evidence("c49_g3.json.gz")["roots"]}
    assert len(g4) == 409 and all(r["complete"] for r in g4)
    assert round(sum(r["build_seconds"] for r in g4) / 3600, 1) == 55.2
    assert round(sum(g3[r["index"]]["build_seconds"] for r in g4) / 3600, 1) == 67.0
    assert max(r["improved"]["complete_seconds"] for r in g4) < 2594
    assert max(r["build_seconds"] for r in g4) < 5171
    a, earlier = _assembly(), [_evidence(name) for name in EARLIER]
    rows = [r for run in earlier for r in run["roots"]]
    before, control = v2_assemble(rows, a)
    library, shared = g4_library(a["library"], _evidence("c49_g3.json.gz"), earlier, SPEC), len(a["library"])
    assert (len(library), shared) == (2530, 2030)
    cost = {**a["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, SPEC["builder"]["level_bits"])
                                for i, (m, empty) in enumerate(library, 1) if i > shared}}
    new = {_key(r): j1_piece(r, library) for r in g4}
    assert len([k for k, p in new.items() if p[0] < a["base"][k][0]]) == 86
    assert len([p for p in new.values() if any(m > shared for m in p[1])]) == 204
    assert (sum(p[0] for p in new.values()), sum(a["base"][k][0] for k in new)) == (55_928, 61_427)
    built = {k: j1_piece(rows[n], a["library"]) for k, n in before.items()}

    def size(b):
        return j1_bits(a["opening"], a["below"], b, built, cost)["bits"]

    base = {**a["base"], **new}
    assert (control, size(base)) == (192_591, 195_394)
    held = set(new)
    while True:
        best = size(base)
        trial = min(((size({**base, k: a["base"][k]}), k) for k in sorted(held)), default=(best, None))
        if trial[0] >= best:
            break
        base[trial[1]] = a["base"][trial[1]]
        held.discard(trial[1])
    assert (size(base), len(held)) == (184_565, 76)
    chosen, bits = v2_assemble(rows, {**a, "base": base, "map_cost": cost})
    index = {k: i for i, k in enumerate(sorted(a["below"]))}
    assert bits == 184_556 and set(chosen) == set(before)
    assert [(index[k], rows[before[k]].get("root_move"), rows[n].get("root_move"),
             j1_piece(rows[before[k]], a["library"])[0], j1_piece(rows[n], a["library"])[0])
            for k, n in chosen.items() if n != before[k]] == [(44, 1, 3, 344, 433)]
    pieces = {k: j1_piece(rows[n], a["library"]) for k, n in chosen.items()}
    assert j1_bits(a["opening"], a["below"], base, pieces, cost) == {
        "bits": 184_556, "kept": 97, "trivial": 123, "strategies": 409, "library_maps": 816}
    kids = {c for k in a["below"] if k not in chosen for c in a["below"][k] if c in base}
    maps = set().union(*(p[1] for p in pieces.values()), *(base[c][1] for c in kids))
    assert len(held & kids) == 76
    assert (sum(base[c][0] for c in held), sum(a["base"][c][0] for c in held)) == (22_674, 29_270)
    assert sum(p[0] for p in pieces.values()) == 57_771 and len(pieces) == 51
    assert sum(base[c][0] for c in kids) == 54_831 and len(kids) == 409
    assert sum(cost[m] for m in maps) == 71_109 and len(maps) == 816
    assert len([m for m in maps if m > shared]) == 83
