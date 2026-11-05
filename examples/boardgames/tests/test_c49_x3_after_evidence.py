"""§3.6 X3 — finding registered AFTER the data, from evidence/c49_x3.json.gz: where the 8,802 bits of X3's whole game
came from, and how little of them needs a mirrored leaf. U1's search stops at 71 of the 409 rebuilds from all taken
(166,623 bits, 0.9498 of 175,425) and at 54 from none taken (167,046); all 409 taken the game takes 176,641. The 409
play 32 mirrored leaves of 2,134, in 18 rebuilds, and those 18 take 6,677 bits against 9,488 for the pieces they stand
in for; the 391 with none take 46,042 against 45,464. With only the 18 as candidates the game takes 172,678, short of
the bar; with only the 391 it takes 169,962, past the bar with no mirrored leaf played. Of the 71 taken 13 read mirrored
and save 2,831 bits, 58 do not and save 3,083. Four builds ran past the 2-hour cap, which ends their improvement, and
three of them are taken. The parts: opening 845, library 70,101 (797 maps, 177 beyond G3's library, 30 of those beyond
G4's), 409 strategies 49,038, builds at 51 ply-6 positions 46,639 (as in X2's game)."""
from __future__ import annotations

import pytest

import harness.floor_x3 as floor_x3
from harness.floor_g4 import SPEC as G4, _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece
from harness.floor_v2 import v2_assemble
from harness.floor_x2 import SPEC as X2, x2_piece
from harness.floor_x3 import SPEC, x3_size
from harness.floor_z1 import SPEC as Z1, z1_library
from harness.steady_state import map_bits
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER, EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _mirrored(row: dict, maps: int) -> int:
    return sum(1 <= n["leaf"] <= maps for _, _, n in row["nodes"] if "leaf" in n)


def test_c49_x3_71_of_the_409_rebuilds_take_the_whole_game_to_166_623_bits(monkeypatch):
    found, real = [], floor_x3.u1_choose
    monkeypatch.setattr(floor_x3, "u1_choose", lambda size, plays, start: found.append((size, real(size, plays, start)))
                        or found[-1][1])
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    a, g3, g4 = _assembly(), _evidence("c49_g3.json.gz"), _evidence("c49_g4.json.gz")
    z, x2 = _evidence("c49_z1.json.gz"), _evidence("c49_x2.json.gz")
    r = x3_size(_evidence(), (x2, X2), (z, Z1), earlier, (g4, G4), g3, a, SPEC)
    assert r == {"verdict": "supported", "bits": 166_623, "control_bits": 175_425, "ratio": 166_623 / 175_425,
                 "rebuilt": 71, "unfinished": False}
    (size, from_none), (_, from_all) = found
    assert (len(from_none[0]), from_none[1], len(from_all[0]), from_all[1]) == (54, 167_046, 71, 166_623)
    held = from_all[0]
    rows = [x for run, _ in earlier for x in run["roots"]]
    before, _ = v2_assemble(rows, a)
    g4_lib = g4_library(a["library"], g3, [run for run, _ in earlier], G4)
    library = z1_library(g4_lib, g4, SPEC)
    shared = len(a["library"])
    cost = {**a["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, SPEC["builder"]["level_bits"])
                                for i, (m, empty) in enumerate(library, 1) if i > shared}}
    base = {**a["base"], **{_key(x): j1_piece(x, g4_lib) for x in g4["roots"] if x["index"] in SPEC["kept"]}}
    built = {**{k: j1_piece(rows[n], a["library"]) for k, n in before.items()},
             **{_key(x): j1_piece(x, library) for x in z["roots"] if x["index"] in SPEC["z_held"]},
             **{_key(x): x2_piece(x, library) for x in x2["roots"] if x["index"] in SPEC["x2_held"]}}
    roots = _evidence()["roots"]
    rebuilds = {_key(x): x2_piece(x, library) for x in roots}
    assert size(frozenset(rebuilds)) == 176_641
    reads = {_key(x): _mirrored(x, len(library)) for x in roots}
    leaves = sum("leaf" in n for x in roots for _, _, n in x["nodes"])
    assert (sum(reads.values()), leaves, sum(n > 0 for n in reads.values())) == (32, 2_134, 18)
    with_reads = [k for k, n in reads.items() if n]
    without = [k for k, n in reads.items() if not n]
    assert (sum(rebuilds[k][0] for k in with_reads), sum(base[k][0] for k in with_reads)) == (6_677, 9_488)
    assert (sum(rebuilds[k][0] for k in without), sum(base[k][0] for k in without)) == (46_042, 45_464)

    def alone(keys):
        plays = {k: rebuilds[k][1] for k in keys}
        chosen, bits = min((real(size, plays, frozenset()), real(size, plays, frozenset(plays))), key=lambda s: s[1])
        return len(chosen), bits

    assert alone(with_reads) == (10, 172_678) and alone(without) == (54, 169_962)
    saved = {k: base[k][0] - rebuilds[k][0] for k in held}
    assert (sum(k in held for k in with_reads), sum(saved[k] for k in with_reads if k in held)) == (13, 2_831)
    assert (sum(k in held for k in without), sum(saved[k] for k in without if k in held)) == (58, 3_083)
    capped = {x["index"]: _key(x) in held for x in roots if x["build_seconds"] >= SPEC["builder"]["seconds"]}
    assert capped == {267: True, 269: True, 273: True, 275: False}
    pieces = {**base, **{k: rebuilds[k] for k in held}}
    assert j1_bits(a["opening"], a["below"], pieces, built, cost) == {
        "bits": 166_623, "kept": 97, "trivial": 123, "strategies": 409, "library_maps": 797}
    kids = {c for k in a["below"] if k not in built for c in a["below"][k] if c in pieces}
    maps = set().union(*(p[1] for p in built.values()), *(pieces[c][1] for c in kids))
    assert sum(p[0] for p in built.values()) == 46_639 and len(built) == 51
    assert sum(pieces[c][0] for c in kids) == 49_038 and len(kids) == 409
    assert sum(cost[m] for m in maps) == 70_101 and len(maps) == 797
    assert (len([m for m in maps if m > shared]), len([m for m in maps if m > len(g4_lib)])) == (177, 30)
