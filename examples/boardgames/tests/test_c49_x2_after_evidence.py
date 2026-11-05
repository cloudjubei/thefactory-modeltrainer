"""§3.6 X2 — finding registered AFTER the data, from evidence/c49_x2.json.gz: where the 4,774 bits of X2's whole game
came from. U1's search stops at the same 7 of the 51 rebuilds from none taken and from all taken, and the whole game
takes 175,425 bits (0.9735 of 180,199); all 51 taken it would take 184,651. Of Z1's 27 rebuilds 22 stay. The 51 play
38 mirrored leaves of 1,773, in 10 rebuilds, and those 10 take 20,260 bits against Z1's 25,085 for the same positions;
the 41 with none take 26,778 against 25,980. Two of the 7 taken read mirrored, #2 (8 leaves) and #4 (7), and save 3,311
and 645 bits on their builds. The parts: opening 845, library 72,989 (832 maps, 151 beyond G3's library, 20 of those
beyond G4's), 409 strategies 54,952 (as in Z1's game), builds at 51 ply-6 positions 46,639."""
from __future__ import annotations

import pytest

import harness.floor_x2 as floor_x2
from harness.floor_g4 import SPEC as G4, _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece
from harness.floor_v2 import v2_assemble
from harness.floor_x2 import SPEC, x2_piece, x2_size
from harness.floor_z1 import SPEC as Z1, z1_library
from harness.steady_state import map_bits
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x2_evidence import EARLIER, EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _mirrored(row: dict, maps: int) -> int:
    return sum(1 <= n["leaf"] <= maps for _, _, n in row["nodes"] if "leaf" in n)


def test_c49_x2_7_of_the_51_rebuilds_take_the_whole_game_to_175_425_bits(monkeypatch):
    found, real = [], floor_x2.u1_choose
    monkeypatch.setattr(floor_x2, "u1_choose", lambda size, plays, start: found.append((size, real(size, plays, start)))
                        or found[-1][1])
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    a, g3, g4, z = _assembly(), _evidence("c49_g3.json.gz"), _evidence("c49_g4.json.gz"), _evidence("c49_z1.json.gz")
    r = x2_size(_evidence(), (z, Z1), earlier, (g4, G4), g3, a, SPEC)
    assert r == {"verdict": "inconclusive", "bits": 175_425, "control_bits": 180_199, "ratio": 175_425 / 180_199,
                 "rebuilt": 7, "unfinished": False}
    (size, from_none), (_, from_all) = found
    assert from_none == from_all and from_none[1] == 175_425
    held = from_none[0]
    rows = [x for run, _ in earlier for x in run["roots"]]
    before, _ = v2_assemble(rows, a)
    g4_lib = g4_library(a["library"], g3, [run for run, _ in earlier], G4)
    library = z1_library(g4_lib, g4, SPEC)
    shared = len(a["library"])
    cost = {**a["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, SPEC["builder"]["level_bits"])
                                for i, (m, empty) in enumerate(library, 1) if i > shared}}
    base = {**a["base"], **{_key(x): j1_piece(x, g4_lib) for x in g4["roots"] if x["index"] in SPEC["kept"]}}
    built = {**{k: j1_piece(rows[n], a["library"]) for k, n in before.items()},
             **{_key(x): j1_piece(x, library) for x in z["roots"] if x["index"] in SPEC["z_held"]}}
    roots = _evidence()["roots"]
    rebuilds = {_key(x): x2_piece(x, library) for x in roots}
    assert size(frozenset(rebuilds)) == 184_651
    taken = {x["index"] for x in roots if _key(x) in held}
    assert taken == {2, 4, 68, 71, 72, 118, 130} and len(set(SPEC["z_held"]) - taken) == 22
    reads = {x["index"]: _mirrored(x, len(library)) for x in roots}
    leaves = sum("leaf" in n for x in roots for _, _, n in x["nodes"])
    assert (sum(reads.values()), leaves, sum(n > 0 for n in reads.values())) == (38, 1_773, 10)
    assert {i: n for i, n in reads.items() if i in taken and n} == {2: 8, 4: 7}
    z_bits = {_key(x): j1_piece(x, library)[0] for x in z["roots"]}
    with_reads = [_key(x) for x in roots if reads[x["index"]]]
    without = [_key(x) for x in roots if not reads[x["index"]]]
    assert (sum(rebuilds[k][0] for k in with_reads), sum(z_bits[k] for k in with_reads)) == (20_260, 25_085)
    assert (sum(rebuilds[k][0] for k in without), sum(z_bits[k] for k in without)) == (26_778, 25_980)
    saved = {x["index"]: built[_key(x)][0] - rebuilds[_key(x)][0] for x in roots}
    assert (saved[2], saved[4]) == (3_311, 645)
    pieces = {**built, **{k: rebuilds[k] for k in held}}
    assert j1_bits(a["opening"], a["below"], base, pieces, cost) == {
        "bits": 175_425, "kept": 97, "trivial": 123, "strategies": 409, "library_maps": 832}
    kids = {c for k in a["below"] if k not in pieces for c in a["below"][k] if c in base}
    maps = set().union(*(p[1] for p in pieces.values()), *(base[c][1] for c in kids))
    assert sum(p[0] for p in pieces.values()) == 46_639 and len(pieces) == 51
    assert sum(base[c][0] for c in kids) == 54_952 and len(kids) == 409
    assert sum(cost[m] for m in maps) == 72_989 and len(maps) == 832
    assert (len([m for m in maps if m > shared]), len([m for m in maps if m > len(g4_lib)])) == (151, 20)
