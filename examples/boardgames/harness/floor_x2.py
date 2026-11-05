"""§3.6 X2 — the whole game's ply-6 builds rebuilt with every map also read on the mirrored board. A map is
cell-indexed and its rule breaks ties by the lowest move, so WeakC4's one diagram per mirror pair has no equal here:
none of G3's maps mirrors another (h335), and a map with its cells flipped covers none of the 343 move nodes of Z1's
51 rebuilds, while the same maps read on the mirrored board cover 18 of them. X2 rebuilds Z1's 51, each at the same
position with the same root move, with the same builder, rule, pipeline and 6-hour cap and from the same 2,598 maps,
with `mirror` on (harness.mirror_builder): the library comes twice (`x2_library`, written by
scripts/c4_library_file.py to `library_from`), each map first as its mirrored reading, then as its plain one. A
mirrored reading plays a stored map, so both readings of a map pay its bits once; a leaf's reference says which it
reads. The whole game is Z1's (180,199 bits, the 27 rebuilds it took standing as builds) with every complete mirror
rebuild a candidate for its position, chosen by U1's search (harness.floor_u1) from none taken and from all taken —
never worse than 180,199. Judged on completion out of 51 (48 supports, 40 or fewer refutes) and on the whole game
against 180,199: supported at 97/100 of it or less, refuted at 99/100 or more once every rebuild finished."""
from __future__ import annotations

from harness.floor_e3 import _root_move
from harness.floor_g4 import _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece, j1_valid
from harness.floor_u1 import u1_choose
from harness.floor_v2 import _order, v2_assemble
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.floor_z1 import LIBRARY_SOURCES, SPEC as Z1, z1_library, z1_library_from
from harness.steady_state import map_bits

SPEC = {**Z1, "measurement_fp": "8d1b222ed933", "library_from": "c49_x2_library.json.gz",
        "builder": {**Z1["builder"], "mirror": True},
        "z_held": [
                    44, 47, 48, 50, 52, 58, 67, 68, 69, 70, 71, 72, 77, 78, 80, 92, 97, 98, 103, 105, 110, 112, 113, 118,
                    124, 130, 143],
        "control_bits": 180_199}

x2_report = w6_report


def x2_library(library: list) -> list:
    return [*library, *library]


def x2_library_from(*sources: dict) -> list:
    return x2_library(z1_library_from(*sources))


def x2_piece(r: dict, library: list) -> tuple:
    """A rebuild's bits and the maps it plays, each map one id whichever way it is read; `library` is the maps once."""
    bits, leaves = j1_piece(r, x2_library(library))
    return bits, frozenset(i - len(library) if i > len(library) else i for i in leaves)


def x2_size(e: dict, z: tuple, earlier: list, g4: tuple, g3: dict, assembly: dict, spec: dict) -> dict:
    """`z` is (Z1's run, its SPEC), `g4` (G4's run, its SPEC) and `earlier` [(run, its SPEC), ...]; `assembly` holds J1's
    opening, below, base, map_cost and G3's library."""
    (z_run, z_spec), (g4_run, g4_spec) = z, g4
    if (_integrity(e, spec) or _integrity(z_run, z_spec) or _integrity(g4_run, g4_spec)
            or any(_integrity(run, s) for run, s in earlier)):
        return {"verdict": "not_run"}
    rows = [r for run, _ in earlier for r in run["roots"]]
    rank = {k: i for i, k in enumerate(sorted(assembly["below"]))}
    if any(rank.get(_key(r)) != r["index"] for r in rows + z_run["roots"] + e["roots"]):
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
    held = [r for r in z_run["roots"] if r["index"] in spec["z_held"]]
    if sorted(r["index"] for r in held) != spec["z_held"] or not all(j1_valid(r, library) for r in held):
        return {"verdict": "not_run"}
    done = [r for r in e["roots"] if r["complete"]]
    if not all(j1_valid(r, x2_library(library)) for r in done):
        return {"verdict": "not_run"}
    level_bits = spec["builder"]["level_bits"]
    cost = {**assembly["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, level_bits)
                                       for i, (m, empty) in enumerate(library, 1) if i > len(assembly["library"])}}
    base = {**assembly["base"], **{_key(r): j1_piece(r, g4_lib) for r in kept}}
    built = {**{k: j1_piece(rows[n], assembly["library"]) for k, n in before.items()},
             **{_key(r): j1_piece(r, library) for r in held}}
    rebuilds = {_key(r): x2_piece(r, library) for r in done}

    def size(chosen):
        return j1_bits(assembly["opening"], assembly["below"], base, {**built, **{k: rebuilds[k] for k in chosen}},
                       cost)["bits"]

    control = size(frozenset())
    if control != spec["control_bits"]:
        return {"verdict": "not_run"}
    plays = {k: piece[1] for k, piece in rebuilds.items()}
    chosen, bits = min((u1_choose(size, plays, frozenset()), u1_choose(size, plays, frozenset(rebuilds))),
                       key=lambda s: s[1])
    unfinished = len(done) < len(e["roots"])
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if not unfinished and rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control, "rebuilt": len(chosen),
            "unfinished": unfinished}
