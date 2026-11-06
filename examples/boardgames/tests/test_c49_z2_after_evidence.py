"""§3.6 Z2 — finding registered AFTER the data, from the evidence of the runs up to Z2: what the 50 rebuilds did. With
each an option for U2's search the whole game stops at 162,752 bits from U2's game and at 163,104 from every position's
smallest piece, 2,271 under U2's 165,023 (0.9862 of it). 21 of the 50 rebuilds are smaller than the piece the game took
at their position and 29 larger, and together they take 42,528 bits against 43,653. The search takes 9: 8 of the smaller
and #118, which is larger. 13 positions differ from U2's game, all in play: those 9 builds (5 of Z1's, 3 earlier, 1 of
X2's), 3 of G3's strategies for G4's and 1 of X3's for G4's. The parts: opening 849, library 70,435 (801 maps, 223
beyond G3's library, 40 of those beyond G4's, 7 of those beyond X2's), 416 strategies 49,692, 50 builds 41,776. No build
reached the 6-hour cap, and the 50 took 47.2 build-hours."""
from __future__ import annotations

from collections import Counter

import pytest

import harness.floor_z2 as floor_z2
from harness.floor_g4 import SPEC as G4, _key, g4_library
from harness.floor_j1 import j1_bits
from harness.floor_u2 import SPEC as U2
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC as X4
from harness.floor_z1 import SPEC as Z1, z1_library
from harness.floor_z2 import SPEC, _runs, z2_library, z2_size
from harness.steady_state import map_bits
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER
from tests.test_c49_z2_evidence import EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _run(label) -> str:
    return "earlier" if isinstance(label, int) else label


def test_c49_z2_the_rebuilds_chosen_again_take_the_whole_game_to_162_752_bits(monkeypatch):
    found, real = [], floor_z2.u2_choose
    monkeypatch.setattr(floor_z2, "u2_choose", lambda size, options, start: found.append(
        (options, start, real(size, options, start))) or found[-1][2])
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    a, g3, g4 = _assembly(), _evidence("c49_g3.json.gz"), _evidence("c49_g4.json.gz")
    z, x2 = _evidence("c49_z1.json.gz"), _evidence("c49_x2.json.gz")
    x3, x4 = _evidence("c49_x3.json.gz"), _evidence("c49_x4.json.gz")
    r = z2_size(_evidence(), (x4, X4), (x3, X3), (x2, X2), (z, Z1), earlier, (g4, G4), g3, a, U2, SPEC)
    state = r.pop("state")
    assert r == {"verdict": "supported", "bits": 162_752, "control_bits": 165_023, "ratio": 162_752 / 165_023,
                 "rebuilt": 9, "unfinished": False}
    assert [bits for _, _, (_, bits) in found] == [162_752, 163_104]
    options, start, _ = found[0]
    below = a["below"]
    g4_lib = g4_library(a["library"], g3, [run for run, _ in earlier], G4)
    x2_lib = z1_library(g4_lib, g4, X3)
    rows = [row for run, _ in earlier for row in run["roots"]]
    library = z2_library(x2_lib, _runs(rows, g3, g4, z, x2, x3, x4), SPEC)
    cost = {**a["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, SPEC["builder"]["level_bits"])
                                for i, (m, empty) in enumerate(library, 1) if i > len(a["library"])}}
    built = {k: options[k][label] for k, label in state.items() if k in below and label is not None}
    base = {**a["base"], **{k: options[k][label] for k, label in state.items() if k not in below}}
    assert j1_bits(a["opening"], below, base, built, cost) == {
        "bits": 162_752, "kept": 98, "trivial": 123, "strategies": 416, "library_maps": 801}
    kids = {c for k in below if k not in built for c in below[k]}
    strategies = {c: base[c] for c in kids if c in base}
    maps = set().union(*(p[1] for p in built.values()), *(p[1] for p in strategies.values()))
    assert sum(p[0] for p in built.values()) == 41_776 and len(built) == 50
    assert sum(p[0] for p in strategies.values()) == 49_692 and len(strategies) == 416
    assert sum(cost[m] for m in maps) == 70_435 and len(maps) == 801
    beyond = [len([m for m in maps if m > len(held)]) for held in (a["library"], g4_lib, x2_lib)]
    assert beyond == [223, 40, 7]
    assert Counter(state[c] for c in strategies) == {"g3": 289, "g4": 76, "x3": 48, "x4": 3}
    assert Counter(_run(state[k]) for k in below) == {None: 98, "z": 20, "earlier": 12, "x2": 9, "z2": 9}
    moved = [k for k in state if state[k] != start[k]]
    assert all(k in below or k in kids for k in moved)
    assert Counter((_run(start[k]), _run(state[k])) for k in moved) == {
        ("z", "z2"): 5, ("earlier", "z2"): 3, ("g3", "g4"): 3, ("x2", "z2"): 1, ("x3", "g4"): 1}
    pieces = [(r["index"], _key(r)) for r in _evidence()["roots"]]
    sizes = {i: (options[k]["z2"][0], options[k][start[k]][0]) for i, k in pieces}
    assert sum(now for now, _ in sizes.values()) == 42_528 and sum(then for _, then in sizes.values()) == 43_653
    smaller = {i for i, (now, then) in sizes.items() if now < then}
    assert len(smaller) == 21 and not any(now == then for now, then in sizes.values())
    taken = {i for i, k in pieces if state[k] == "z2"}
    assert taken == {44, 47, 48, 79, 89, 103, 114, 118, 140} and taken - smaller == {118}
    seconds = [r["build_seconds"] for r in _evidence()["roots"]]
    assert max(seconds) < SPEC["builder"]["seconds"] and round(sum(seconds) / 3600, 1) == 47.2
