"""§3.6 Z2 — the ply-6 builds of U2's game rebuilt from every map the game holds. U2 chose the whole game again from
every certified piece the runs hold (165,023 bits; h373), and its 50 builds take 43,653 of them. Those builds were built
from at most X2's 2,598 maps, so a map a piece found itself is paid inline by that piece and reachable to no build. Z2
rebuilds the 50, each at the same position with the same root move as the piece the game takes (none where that piece
had none), with X2's builder, rule, pipeline and 6-hour cap and both readings, from X2's maps then the own maps of the
pieces the game plays that the list does not hold yet (`z2_library`; `own` names those pieces, the judge recomputes them
from U2's game, and scripts/c4_library_file.py writes the list twice to `library_from`). The ply-8 strategies are not
rebuilt. Every complete rebuild is one more option at its position for U2's search (harness.floor_u2), from U2's game
and from every position's smallest piece — never worse than 165,023. Judged on completion out of 50 (47 supports, 39
or fewer refutes) and on the whole game against 165,023: supported at 99/100 of it or less, refuted at 999/1000 or
more once every rebuild finished."""
from __future__ import annotations

from harness.floor_e3 import _root_move
from harness.floor_g4 import _key, g4_library
from harness.floor_j1 import j1_bits, j1_valid
from harness.floor_u2 import _smallest, u2_choose, u2_game, u2_size
from harness.floor_v2 import _order
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.floor_x2 import SPEC as X2, x2_library, x2_piece
from harness.floor_z1 import LIBRARY_SOURCES as Z1_SOURCES, z1_library, z1_library_from
from harness.steady_state import map_bits

SPEC = {**{k: v for k, v in X2.items() if k not in ("kept", "z_held")}, "library_from": "c49_z2_library.json.gz",
        "roots": 50,
        "indices": [
                    0, 1, 2, 4, 14, 18, 21, 28, 42, 44, 47, 48, 50, 52, 54, 58, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80,
                    81, 87, 89, 91, 92, 97, 98, 103, 105, 110, 112, 113, 114, 115, 118, 124, 125, 126, 130, 135, 140,
                    143, 145, 146],
        "root_moves": [
                    [0, 4], [1, 2], [2, None], [4, 5], [14, None], [18, 4], [21, None], [28, None], [42, None], [44, 1],
                    [47, 3], [48, None], [50, 3], [52, None], [54, 3], [58, None], [67, 3], [68, None], [69, 1],
                    [70, 0], [71, 2], [72, 5], [77, None], [78, None], [79, 4], [80, 2], [81, None], [87, 1],
                    [89, None], [91, 0], [92, None], [97, None], [98, 3], [103, None], [105, None], [110, 3], [112, 3],
                    [113, None], [114, 1], [115, None], [118, 3], [124, None], [125, None], [126, None], [130, None],
                    [135, None], [140, 1], [143, 1], [145, None], [146, 4]],
        "own": {
            "earlier": [4, 6, 10, 26, 34, 35, 39, 42, 115, 141, 173],
            "g3": [15, 29, 41, 43, 44, 64, 66, 78, 79, 80, 81, 82, 91, 95, 102, 113, 135, 168, 176, 244, 276, 277, 279,
                   302, 337, 400, 401, 407, 457, 490, 495, 505, 518, 529, 563, 569, 604, 609, 638],
            "g4": [25, 28, 40, 54, 56, 86, 110, 185, 226, 227, 260, 265, 275, 358, 361, 397, 409, 411, 466, 476, 493,
                   530, 575],
            "z": [48, 58, 67, 69, 91, 92, 97, 103, 110, 140, 143],
            "x2": [1, 2, 4, 68, 71, 72, 81, 146],
            "x3": [63, 88, 114, 122, 267, 269, 273, 280, 305, 344, 376, 428, 436, 473, 491, 494, 496, 498, 509, 565,
                   570],
            "x4": [57, 335, 499]},
        "complete_support": 47, "complete_refute": 39, "support": (99, 100), "refute": (999, 1000),
        "control_bits": 165_023}
LIBRARY_SOURCES = (*Z1_SOURCES, "c49_z1.json.gz", "c49_x2.json.gz", "c49_x3.json.gz", "c49_x4.json.gz")

z2_report = w6_report


def _runs(rows: list, *runs: dict) -> dict:
    """`rows` are the earlier builds, labelled as U2 labels them; `runs` G3's, G4's, Z1's, X2's, X3's and X4's."""
    return {"earlier": dict(enumerate(rows)), **{name: {r["index"]: r for r in run["roots"]}
                                                 for name, run in zip(("g3", "g4", "z", "x2", "x3", "x4"), runs)}}


def _taken(state: dict, runs: dict, below: dict) -> list:
    kids = {c for k in below if state[k] is None for c in below[k]}
    return [(name, label, r) for name, rows in runs.items() for label, r in rows.items()
            if state.get(_key(r)) == (label if name == "earlier" else name) and (_key(r) in below or _key(r) in kids)]


def z2_library(library: list, runs: dict, spec: dict) -> list:
    """`library` is X2's, once; `runs` maps a run's name to its rows by label."""
    held = [list(m) for m in library]
    for name, labels in spec["own"].items():
        for label in labels:
            held += [m for m in runs[name][label]["found_maps"] if m not in held]
    return held


def z2_library_from(*sources: dict) -> list:
    *first, z, x2, x3, x4 = sources
    g3, earlier, g4 = first[2], first[3:-1], first[-1]
    rows = [r for run in earlier for r in run["roots"]]
    return x2_library(z2_library(z1_library_from(*first), _runs(rows, g3, g4, z, x2, x3, x4), SPEC))


def z2_size(e: dict, x4: tuple, x3: tuple, x2: tuple, z: tuple, earlier: list, g4: tuple, g3: dict, assembly: dict,
            u2_spec: dict, spec: dict) -> dict:
    """Each run comes as (run, its SPEC), `earlier` as a list of them, as `u2_size` takes them; `u2_size` giving the
    registered bits is what proves the earlier runs and gives the game."""
    judged = (x4, x3, x2, z, earlier, g4, g3, assembly, u2_spec)
    state = u2_size(*judged).get("state")
    if _integrity(e, spec) or state is None:
        return {"verdict": "not_run"}
    below = assembly["below"]
    rows = [r for run, _ in earlier for r in run["roots"]]
    runs = _runs(rows, g3, g4[0], z[0], x2[0], x3[0], x4[0])
    taken = _taken(state, runs, below)
    builds = sorted(([r["index"], r.get("root_move")] for _, _, r in taken if _key(r) in below), key=_order)
    own = {name: sorted(label for n, label, r in taken if n == name and r["found_maps"]) for name in runs}
    rank = {k: i for i, k in enumerate(sorted(below))}
    if (builds != spec["root_moves"] or own != spec["own"] or any(rank.get(_key(r)) != r["index"] for r in e["roots"])
            or sorted(([r["index"], r["root_move"]] for r in e["roots"]), key=_order) != spec["root_moves"]
            or any(r["root_move"] is not None and _root_move(r) != r["root_move"] for r in e["roots"])):
        return {"verdict": "not_run"}
    g4_lib = g4_library(assembly["library"], g3, [run for run, _ in earlier], g4[1])
    library = z2_library(z1_library(g4_lib, g4[0], x3[1]), runs, spec)
    done = [r for r in e["roots"] if r["complete"]]
    if not all(j1_valid(r, x2_library(library)) for r in done):
        return {"verdict": "not_run"}
    level_bits = spec["builder"]["level_bits"]
    cost = {**assembly["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, level_bits)
                                       for i, (m, empty) in enumerate(library, 1) if i > len(assembly["library"])}}
    options = u2_game(*judged)[0]
    for r in done:
        options[_key(r)]["z2"] = x2_piece(r, library)

    def size(choice):
        base = {**assembly["base"], **{k: options[k][label] for k, label in choice.items()}}
        built = {k: options[k][label] for k, label in choice.items() if k in below and label is not None}
        return j1_bits(assembly["opening"], below, base, built, cost)["bits"]

    control = spec["control_bits"]
    if size(state) != control:
        return {"verdict": "not_run"}
    smallest = {k: _smallest(pieces, lambda p: True) for k, pieces in options.items()}
    chosen, bits = min((u2_choose(size, options, state), u2_choose(size, options, smallest)), key=lambda s: s[1])
    unfinished = len(done) < len(e["roots"])
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if not unfinished and rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control,
            "rebuilt": sum(label == "z2" for label in chosen.values()), "unfinished": unfinished, "state": chosen}
