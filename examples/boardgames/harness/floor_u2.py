"""§3.6 U2 — the whole game chosen again from every certified piece the runs hold. Each pass chose only among its own
rebuilds, against the game before it: X3's search kept or sent back X3's rebuilds and stopped at two different games
from its two starts (h370), and a piece an earlier pass sent back for the map it alone paid was never looked at again,
though the game now holds other maps. U2 lets every position take any piece: at ply 8 G3's strategy, G4's, X3's or X4's
rebuild; at ply 6 any earlier build, Z1's or X2's rebuild, or no build. A move changes one position, or moves every
position that can to its smallest piece playing a map, or every position paying a map to its smallest piece without it;
the move that shrinks the whole game most is taken until none does (`u2_choose`). A position keeps a build unless a move
of its own drops it. It starts from X3's game (166,623 bits, recomputed by `x3_size`) and from every position's smallest
piece, and keeps the smaller — never worse than 166,623. No new builds. Judged against 166,623: supported at 199/200
of it or less, refuted at 999/1000 or more."""
from __future__ import annotations

from harness.floor_g4 import _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece, j1_valid
from harness.floor_v2 import v2_assemble
from harness.floor_w1 import _integrity
from harness.floor_x2 import x2_library, x2_piece
from harness.floor_x3 import x3_size
from harness.floor_x4 import _mirrored
from harness.floor_z1 import z1_library
from harness.steady_state import map_bits

SPEC = {"x3_held": [
            15, 23, 41, 55, 63, 66, 71, 75, 88, 99, 104, 111, 114, 116, 122, 123, 131, 133, 134, 145, 146,
            148, 149, 182, 194, 198, 200, 266, 267, 269, 272, 273, 279, 280, 301, 305, 306, 325, 327, 335,
            342, 344, 355, 357, 367, 372, 376, 394, 398, 404, 410, 416, 426, 428, 473, 476, 491, 494, 496,
            498, 504, 505, 509, 554, 565, 570, 581, 585, 594, 626, 641],
        "control_bits": 166_623, "support": (199, 200), "refute": (999, 1000)}


def _smallest(pieces: dict, keep) -> object:
    return min((label for label, piece in pieces.items() if piece and keep(piece)), key=lambda label: pieces[label][0],
               default=None)


def u2_choose(size, options: dict, start: dict) -> tuple:
    """`options` maps a position to {label: piece}, a piece (bits, maps) or None for no piece; `size` gives the whole
    game's bits for {position: label}. Returns (the choice, its bits)."""
    state, best = dict(start), size(start)
    maps = sorted({m for pieces in options.values() for piece in pieces.values() if piece for m in piece[1]})
    while True:
        moves = [{k: label} for k, pieces in options.items() for label in pieces if label != state[k]]
        for m in maps:
            into, out = {}, {}
            for k, pieces in options.items():
                piece = pieces[state[k]]
                if piece and m in piece[1]:
                    label = _smallest(pieces, lambda p: m not in p[1])
                    if label is not None:
                        out[k] = label
                else:
                    label = _smallest(pieces, lambda p: m in p[1])
                    if label is not None:
                        into[k] = label
            moves += [group for group in (into, out) if len(group) > 1]
        trial = min(((size({**state, **move}), n) for n, move in enumerate(moves)), default=(best, None))
        if trial[0] >= best:
            return state, best
        state, best = {**state, **moves[trial[1]]}, trial[0]


def u2_game(x4: tuple, x3: tuple, x2: tuple, z: tuple, earlier: list, g4: tuple, g3: dict, assembly: dict,
            spec: dict) -> tuple | None:
    """Each run comes as (run, its SPEC), `earlier` as a list of them. Returns (options, X3's game as a choice, size),
    or None where a run is off its registration or the game is not X3's."""
    (x4_run, x4_spec), (x3_run, x3_spec), (x2_run, x2_spec) = x4, x3, x2
    (z_run, _), (g4_run, g4_spec) = z, g4
    if (_integrity(x4_run, x4_spec)
            or x3_size(x3_run, x2, z, earlier, g4, g3, assembly, x3_spec).get("bits") != spec["control_bits"]):
        return None
    rows = [r for run, _ in earlier for r in run["roots"]]
    before, _ = v2_assemble(rows, assembly)
    g4_lib = g4_library(assembly["library"], g3, [run for run, _ in earlier], g4_spec)
    library = z1_library(g4_lib, g4_run, x3_spec)
    g4_done = [r for r in g4_run["roots"] if r["complete"]]
    x4_done = [r for r in x4_run["roots"] if r["complete"]]
    if (not all(j1_valid(r, g4_lib) for r in g4_done) or any(_mirrored(r) for r in x4_done)
            or not all(j1_valid(r, x2_library(library)) for r in x4_done)):
        return None
    level_bits = x3_spec["builder"]["level_bits"]
    cost = {**assembly["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, level_bits)
                                       for i, (m, empty) in enumerate(library, 1) if i > len(assembly["library"])}}
    options: dict = {}
    for k in assembly["below"]:
        options.setdefault(k, {})[None] = None
    for n, r in enumerate(rows):
        options.setdefault(_key(r), {})[n] = j1_piece(r, assembly["library"])
    for label, run, price in (("z", z_run, j1_piece), ("x2", x2_run, x2_piece)):
        for r in run["roots"]:
            if r["complete"]:
                options[_key(r)][label] = price(r, library)
    for k, piece in assembly["base"].items():
        options.setdefault(k, {})["g3"] = piece
    for r in g4_done:
        options[_key(r)]["g4"] = j1_piece(r, g4_lib)
    for label, run in (("x3", x3_run), ("x4", x4_run)):
        for r in run["roots"]:
            if r["complete"]:
                options[_key(r)][label] = x2_piece(r, library)
    start = {k: None if k in assembly["below"] else "g3" for k in options}
    start.update(before)
    start.update({_key(r): "g4" for r in g4_run["roots"] if r["index"] in x3_spec["kept"]})
    start.update({_key(r): "z" for r in z_run["roots"] if r["index"] in x3_spec["z_held"]})
    start.update({_key(r): "x2" for r in x2_run["roots"] if r["index"] in x3_spec["x2_held"]})
    start.update({_key(r): "x3" for r in x3_run["roots"] if r["index"] in spec["x3_held"]})

    def size(state):
        below = assembly["below"]
        base = {**assembly["base"], **{k: options[k][label] for k, label in state.items() if k not in below}}
        built = {k: options[k][label] for k, label in state.items() if k in below and label is not None}
        return j1_bits(assembly["opening"], assembly["below"], base, built, cost)["bits"]

    return options, start, size


def u2_size(x4: tuple, x3: tuple, x2: tuple, z: tuple, earlier: list, g4: tuple, g3: dict, assembly: dict,
            spec: dict) -> dict:
    game = u2_game(x4, x3, x2, z, earlier, g4, g3, assembly, spec)
    if game is None or game[2](game[1]) != spec["control_bits"]:
        return {"verdict": "not_run"}
    options, start, size = game
    smallest = {k: _smallest(pieces, lambda p: True) for k, pieces in options.items()}
    state, bits = min((u2_choose(size, options, start), u2_choose(size, options, smallest)), key=lambda s: s[1])
    control = spec["control_bits"]
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control,
            "changed": sum(state[k] != start[k] for k in state), "state": state}
