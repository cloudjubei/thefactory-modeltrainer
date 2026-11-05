"""§3.6 Z1 — finding registered AFTER the data, from evidence/c49_z1.json.gz: where the 4,129 bits of Z1's whole game
came from. U1's search stops at the same 27 of the 51 rebuilds from none taken and from all taken, and the whole game
takes 180,199 bits (0.9776 of 184,328); all 51 taken it would take 186,419. 36 of the 51 are smaller than their builds
on their own, and together they take 51,065 bits against 57,682. The 27 taken take 21,606 against 28,215. The parts:
opening 845, library 73,329 (837 maps, 129 beyond G3's library, 14 of those beyond G4's), 409 strategies 54,952 (as in
U1's game), builds at 51 ply-6 positions 51,073."""
from __future__ import annotations

import pytest

import harness.floor_z1 as floor_z1
from harness.floor_g4 import SPEC as G4, _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece
from harness.floor_v2 import v2_assemble
from harness.floor_z1 import SPEC, z1_library, z1_size
from harness.steady_state import map_bits
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_z1_evidence import EARLIER, EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_z1_27_of_the_51_rebuilds_take_the_whole_game_to_180_199_bits(monkeypatch):
    found, real = [], floor_z1.u1_choose
    monkeypatch.setattr(floor_z1, "u1_choose", lambda size, plays, start: found.append((size, real(size, plays, start)))
                        or found[-1][1])
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    a, g3, g4 = _assembly(), _evidence("c49_g3.json.gz"), _evidence("c49_g4.json.gz")
    r = z1_size(_evidence(), earlier, (g4, G4), g3, a, SPEC)
    assert r == {"verdict": "inconclusive", "bits": 180_199, "control_bits": 184_328, "ratio": 180_199 / 184_328,
                 "rebuilt": 27, "unfinished": False}
    (size, from_none), (_, from_all) = found
    assert from_none == from_all and len(from_none[0]) == 27 and from_none[1] == 180_199
    held = from_none[0]
    rows = [x for run, _ in earlier for x in run["roots"]]
    before, _ = v2_assemble(rows, a)
    g4_lib = g4_library(a["library"], g3, [run for run, _ in earlier], G4)
    library = z1_library(g4_lib, g4, SPEC)
    shared = len(a["library"])
    cost = {**a["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, SPEC["builder"]["level_bits"])
                                for i, (m, empty) in enumerate(library, 1) if i > shared}}
    base = {**a["base"], **{_key(x): j1_piece(x, g4_lib) for x in g4["roots"] if x["index"] in SPEC["kept"]}}
    built = {k: j1_piece(rows[n], a["library"]) for k, n in before.items()}
    rebuilds = {_key(x): j1_piece(x, library) for x in _evidence()["roots"]}
    assert size(frozenset(rebuilds)) == 186_419
    assert sum(rebuilds[k][0] < built[k][0] for k in rebuilds) == 36
    assert (sum(p[0] for p in rebuilds.values()), sum(p[0] for p in built.values())) == (51_065, 57_682)
    assert (sum(rebuilds[k][0] for k in held), sum(built[k][0] for k in held)) == (21_606, 28_215)
    pieces = {**built, **{k: rebuilds[k] for k in held}}
    assert j1_bits(a["opening"], a["below"], base, pieces, cost) == {
        "bits": 180_199, "kept": 97, "trivial": 123, "strategies": 409, "library_maps": 837}
    kids = {c for k in a["below"] if k not in pieces for c in a["below"][k] if c in base}
    maps = set().union(*(p[1] for p in pieces.values()), *(base[c][1] for c in kids))
    assert sum(p[0] for p in pieces.values()) == 51_073 and len(pieces) == 51
    assert sum(base[c][0] for c in kids) == 54_952 and len(kids) == 409
    assert sum(cost[m] for m in maps) == 73_329 and len(maps) == 837
    assert (len([m for m in maps if m > shared]), len([m for m in maps if m > len(g4_lib)])) == (129, 14)
