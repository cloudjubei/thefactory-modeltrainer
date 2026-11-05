"""§3.6 U2 — finding registered AFTER the data, from the evidence of the runs up to X4: what choosing the whole game
again found. From X3's game U2's search stops at 165,196 bits and from every position's smallest piece at 165,023, the
smaller, 1,600 under X3's 166,623 (0.9904 of it). 49 positions differ from X3's game, all of them in play: 25 of X3's
71 rebuilds leave it (19 for G4's piece, 5 for G3's, 1 for X4's) and 11 of G3's strategies go to G4's; 7 of the 51
builds move, one to no build. Three of X4's mirror-off rebuilds are taken. The parts: opening 849, library 70,917 (807
maps, 222 beyond G3's library, 35 of those beyond G4's), 416 strategies 49,604, 50 builds 43,653."""
from __future__ import annotations

from collections import Counter

import pytest

import harness.floor_u2 as floor_u2
from harness.floor_g4 import SPEC as G4, g4_library
from harness.floor_j1 import j1_bits
from harness.floor_u2 import SPEC, u2_game, u2_size
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC as X4
from harness.floor_z1 import SPEC as Z1, z1_library
from harness.steady_state import map_bits
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER
from tests.test_c49_x4_evidence import EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _run(label) -> str:
    return "earlier" if isinstance(label, int) else label


def test_c49_u2_every_certified_piece_chosen_again_takes_the_whole_game_to_165_023_bits(monkeypatch):
    found, real = [], floor_u2.u2_choose
    monkeypatch.setattr(floor_u2, "u2_choose", lambda size, options, start: found.append(real(size, options, start))
                        or found[-1])
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    a, g3, g4 = _assembly(), _evidence("c49_g3.json.gz"), _evidence("c49_g4.json.gz")
    runs = ((_evidence(), X4), (_evidence("c49_x3.json.gz"), X3), (_evidence("c49_x2.json.gz"), X2),
            (_evidence("c49_z1.json.gz"), Z1), earlier, (g4, G4), g3, a, SPEC)
    r = u2_size(*runs)
    state = r.pop("state")
    assert r == {"verdict": "supported", "bits": 165_023, "control_bits": 166_623, "ratio": 165_023 / 166_623,
                 "changed": 49}
    assert [bits for _, bits in found] == [165_196, 165_023]
    options, start, _ = u2_game(*runs)
    below = a["below"]
    g4_lib = g4_library(a["library"], g3, [run for run, _ in earlier], G4)
    library = z1_library(g4_lib, g4, X3)
    cost = {**a["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, X3["builder"]["level_bits"])
                                for i, (m, empty) in enumerate(library, 1) if i > len(a["library"])}}
    built = {k: options[k][label] for k, label in state.items() if k in below and label is not None}
    base = {**a["base"], **{k: options[k][label] for k, label in state.items() if k not in below}}
    assert j1_bits(a["opening"], below, base, built, cost) == {
        "bits": 165_023, "kept": 98, "trivial": 123, "strategies": 416, "library_maps": 807}
    kids = {c for k in below if k not in built for c in below[k]}
    strategies = {c: base[c] for c in kids if c in base}
    maps = set().union(*(p[1] for p in built.values()), *(p[1] for p in strategies.values()))
    assert sum(p[0] for p in built.values()) == 43_653 and len(built) == 50
    assert sum(p[0] for p in strategies.values()) == 49_604 and len(strategies) == 416
    assert sum(cost[m] for m in maps) == 70_917 and len(maps) == 807
    shared = len(a["library"])
    assert (len([m for m in maps if m > shared]), len([m for m in maps if m > len(g4_lib)])) == (222, 35)
    assert a["opening"]["moves"] * 4 + a["opening"]["trivial"] + 4 * 98 + 123 == 849
    assert Counter(state[c] for c in strategies) == {"g3": 292, "g4": 72, "x3": 49, "x4": 3}
    assert Counter(_run(state[k]) for k in below) == {None: 98, "z": 25, "earlier": 15, "x2": 10}
    moved = [k for k in state if state[k] != start[k]]
    assert all(k in below or k in kids for k in moved)
    assert Counter((_run(start[k]), _run(state[k])) for k in moved) == {
        ("x3", "g4"): 19, ("g3", "g4"): 11, ("x3", "g3"): 5, ("earlier", "x2"): 3, ("earlier", "z"): 3, ("g3", "x4"): 2,
        ("g3", "x3"): 2, ("earlier", None): 1, ("g4", "g3"): 1, ("g4", "x3"): 1, ("x3", "x4"): 1}
