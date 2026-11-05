"""§3.6 U1 — finding registered AFTER the data, from evidence/c49_g4.json.gz: choosing among G4's 409 rebuilds by
local search (harness.floor_u1: one rebuild, or every rebuild playing one map, in or out at a time, the move that
shrinks the whole game most first) instead of G4's sending back one at a time from all in. From none kept the search
stops at 61 rebuilds and 184,410 bits; from all kept at 66 and 184,328. Choosing the ply-6 builds again would take
184,386, so V2's choice stays, and the whole game takes 184,328 bits — 228 fewer than G4's 184,556 (0.9988). The
parts: opening 845, library 70,849 (814 maps, 63 beyond G3's library), 409 strategies 54,952 (the 66 rebuilds 16,901
against G3's 23,376), builds at 51 ply-6 positions 57,682."""
from __future__ import annotations

import pytest

import harness.floor_u1 as floor_u1
from harness.floor_g4 import SPEC, _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece
from harness.floor_u1 import u1_size
from harness.floor_v2 import v2_assemble
from harness.steady_state import map_bits
from tests.test_c49_g4_evidence import EARLIER, EVIDENCE, FILES, LABELS, _evidence
from tests.test_c49_v1_evidence import _assembly

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_u1_local_search_over_g4_s_rebuilds_saves_228_bits(monkeypatch):
    found, real = [], floor_u1.u1_choose
    monkeypatch.setattr(floor_u1, "u1_choose", lambda size, plays, start: found.append(real(size, plays, start))
                        or found[-1])
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    a = _assembly()
    r = u1_size(_evidence(), earlier, _evidence("c49_g3.json.gz"), a, SPEC)
    assert r == {"verdict": "measured", "bits": 184_328, "g4_bits": 184_556, "kept": 66, "from_none": 184_410,
                 "from_all": 184_328}
    assert [len(held) for held, _ in found] == [61, 66]
    rows = [x for run, _ in earlier for x in run["roots"]]
    before, _ = v2_assemble(rows, a)
    library = g4_library(a["library"], _evidence("c49_g3.json.gz"), [run for run, _ in earlier], SPEC)
    shared = len(a["library"])
    cost = {**a["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, SPEC["builder"]["level_bits"])
                                for i, (m, empty) in enumerate(library, 1) if i > shared}}
    held = found[1][0]
    base = {**a["base"], **{_key(x): j1_piece(x, library) for x in _evidence()["roots"] if _key(x) in held}}
    assert v2_assemble(rows, {**a, "base": base, "map_cost": cost})[1] == 184_386
    pieces = {k: j1_piece(rows[n], a["library"]) for k, n in before.items()}
    assert j1_bits(a["opening"], a["below"], base, pieces, cost) == {
        "bits": 184_328, "kept": 97, "trivial": 123, "strategies": 409, "library_maps": 814}
    kids = {c for k in a["below"] if k not in before for c in a["below"][k] if c in base}
    maps = set().union(*(p[1] for p in pieces.values()), *(base[c][1] for c in kids))
    assert sum(p[0] for p in pieces.values()) == 57_682 and len(pieces) == 51
    assert sum(base[c][0] for c in kids) == 54_952 and len(kids) == 409
    assert sum(cost[m] for m in maps) == 70_849 and len(maps) == 814
    assert len([m for m in maps if m > shared]) == 63
    assert (sum(base[c][0] for c in held), sum(a["base"][c][0] for c in held)) == (16_901, 23_376)
