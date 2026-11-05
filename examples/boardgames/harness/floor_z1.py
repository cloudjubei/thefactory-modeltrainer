"""§3.6 Z1 — the assembled whole game's ply-6 builds rebuilt. G4 rebuilt the ply-8 strategies the assembled game keeps
from every map it held, and the whole game took 0.958 of its bits (h358); U1 chose among those rebuilds (184,328 bits;
h360). The 51 strategies built from ply 6 (57,682 bits, 31% of the game) were never rebuilt: each was built from G3's
library and pays every map it found itself inline, so a map one of them found is paid again by any other piece that
would play it. Z1 rebuilds those 51, each at the same position with the same root move as the build the game takes
(none where that build had none), with the builder, rule, pipeline and 6-hour cap they were built with, from every map
the game holds: G4's library in its order, then the own maps of the 66 rebuilds U1 keeps, in position order
(`z1_library`, written by scripts/c4_library_file.py to `library_from`). The ply-8 strategies are not rebuilt. The whole
game is U1's with every complete rebuild a candidate for its build, chosen by U1's search (harness.floor_u1) from none
taken and from all taken — never worse than 184,328. Judged on completion out of 51 (48 supports, 40 or fewer refutes)
and on the whole game against 184,328: supported at 97/100 of it or less, refuted at 99/100 or more once every rebuild
finished."""
from __future__ import annotations

from harness.floor_e3 import _root_move
from harness.floor_g4 import SPEC as G4, _key, g4_library, g4_library_from
from harness.floor_j1 import j1_bits, j1_piece, j1_valid
from harness.floor_u1 import u1_choose
from harness.floor_v2 import SPEC as V2, _order, v2_assemble
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.steady_state import map_bits

SPEC = {**{k: v for k, v in V2.items() if k not in ("scope", "min_group_bits")},
        "library_from": "c49_z1_library.json.gz", "earlier": G4["earlier"], "roots": 51,
        "indices": [
                    0, 1, 2, 3, 4, 14, 18, 21, 28, 42, 44, 47, 48, 50, 52, 54, 58, 67, 68, 69, 70, 71, 72, 77, 78, 79,
                    80, 81, 87, 89, 91, 92, 97, 98, 103, 105, 110, 112, 113, 114, 115, 118, 124, 125, 126, 130, 135,
                    140, 143, 145, 146],
        "root_moves": [
                    [0, 4], [1, 2], [2, None], [3, 1], [4, 5], [14, None], [18, 4], [21, None], [28, None], [42, None],
                    [44, 1], [47, 3], [48, None], [50, 3], [52, None], [54, 3], [58, None], [67, 3], [68, None], [69,
                    1], [70, 0], [71, 2], [72, 5], [77, None], [78, None], [79, 4], [80, 2], [81, None], [87, 1], [89,
                    None], [91, 0], [92, None], [97, None], [98, 3], [103, None], [105, None], [110, 3], [112, 3], [113,
                    None], [114, 1], [115, None], [118, 3], [124, None], [125, None], [126, None], [130, None], [135,
                    None], [140, 1], [143, 1], [145, None], [146, 4]],
        "kept": [
                    21, 25, 28, 36, 38, 40, 55, 75, 86, 94, 95, 99, 104, 110, 112, 122, 125, 139, 178, 182, 184, 185,
                    200, 265, 266, 269, 275, 278, 279, 287, 301, 305, 306, 333, 344, 348, 358, 361, 372, 380, 381, 386,
                    394, 397, 402, 411, 442, 466, 468, 476, 482, 486, 493, 494, 498, 508, 528, 530, 535, 570, 575, 583,
                    593, 621, 626, 641],
        "complete_support": 48, "complete_refute": 40, "support": (97, 100), "refute": (99, 100),
        "control_bits": 184_328}
LIBRARY_SOURCES = ("c49_g2.json.gz", "c49_f2.json.gz", "c49_g3.json.gz", *SPEC["earlier"], "c49_g4.json.gz")

z1_report = w6_report


def z1_library(library: list, g4: dict, spec: dict) -> list:
    """`library` is G4's and `g4` its run."""
    own = {r["index"]: r["found_maps"] for r in g4["roots"]}
    return [*library] + [m for i in spec["kept"] for m in own[i]]


def z1_library_from(g2: dict, f2: dict, g3: dict, *runs: dict) -> list:
    return z1_library(g4_library_from(g2, f2, g3, *runs[:-1]), runs[-1], SPEC)


def z1_size(e: dict, earlier: list, g4: tuple, g3: dict, assembly: dict, spec: dict) -> dict:
    """`earlier` is [(run, its SPEC), ...] and `g4` (G4's run, its SPEC); `assembly` holds J1's opening, below, base,
    map_cost and G3's library."""
    g4_run, g4_spec = g4
    if _integrity(e, spec) or _integrity(g4_run, g4_spec) or any(_integrity(run, s) for run, s in earlier):
        return {"verdict": "not_run"}
    rows = [r for run, _ in earlier for r in run["roots"]]
    rank = {k: i for i, k in enumerate(sorted(assembly["below"]))}
    if any(rank.get(_key(r)) != r["index"] for r in rows + e["roots"]):
        return {"verdict": "not_run"}
    if not all(j1_valid(r, assembly["library"]) for r in rows):
        return {"verdict": "not_run"}
    before, _ = v2_assemble(rows, assembly)
    taken = [[rows[n]["index"], rows[n].get("root_move")] for n in before.values()]
    if sorted(taken, key=_order) != spec["root_moves"]:
        return {"verdict": "not_run"}
    if (sorted(([r["index"], r["root_move"]] for r in e["roots"]), key=_order) != spec["root_moves"]
            or any(r["root_move"] is not None and _root_move(r) != r["root_move"] for r in e["roots"])):
        return {"verdict": "not_run"}
    g4_lib = g4_library(assembly["library"], g3, [run for run, _ in earlier], g4_spec)
    kept = [r for r in g4_run["roots"] if r["index"] in spec["kept"]]
    if sorted(r["index"] for r in kept) != spec["kept"] or not all(j1_valid(r, g4_lib) for r in kept):
        return {"verdict": "not_run"}
    library = z1_library(g4_lib, g4_run, spec)
    done = [r for r in e["roots"] if r["complete"]]
    if not all(j1_valid(r, library) for r in done):
        return {"verdict": "not_run"}
    level_bits = spec["builder"]["level_bits"]
    cost = {**assembly["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, level_bits)
                                       for i, (m, empty) in enumerate(library, 1) if i > len(assembly["library"])}}
    base = {**assembly["base"], **{_key(r): j1_piece(r, g4_lib) for r in kept}}
    built = {k: j1_piece(rows[n], assembly["library"]) for k, n in before.items()}
    rebuilds = {_key(r): j1_piece(r, library) for r in done}

    def size(held):
        return j1_bits(assembly["opening"], assembly["below"], base, {**built, **{k: rebuilds[k] for k in held}},
                       cost)["bits"]

    control = size(frozenset())
    if control != spec["control_bits"]:
        return {"verdict": "not_run"}
    plays = {k: piece[1] for k, piece in rebuilds.items()}
    held, bits = min((u1_choose(size, plays, frozenset()), u1_choose(size, plays, frozenset(rebuilds))),
                     key=lambda s: s[1])
    unfinished = len(done) < len(e["roots"])
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if not unfinished and rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control, "rebuilt": len(held),
            "unfinished": unfinished}
