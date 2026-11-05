"""§3.6 X3 — the whole game's ply-8 strategies rebuilt with every map also read on the mirrored board. X2 gave the 51
ply-6 builds both readings of every map, and the 10 rebuilds playing a mirrored leaf took 20,260 bits against Z1's
25,085 (h367); the 409 ply-8 strategies (54,952 bits) were not rebuilt with it. X3 rebuilds those 409, the positions G4
rebuilt, with G4's builder, rule, pipeline and cap, with `mirror` on (harness.mirror_builder), from the same 2,598 maps
read both ways as X2 (its `library_from`). The pieces they stand in for were built from smaller libraries, so a rebuild
differs from its piece by more than the second reading (h370). The ply-6 builds are not rebuilt. The whole game is X2's
(175,425 bits, recomputed by `x2_size`), with every complete rebuild a candidate for its position, chosen by U1's search
(harness.floor_u1) from none taken and from all taken — never worse than 175,425. Judged on completion out of 409 (all
supports, 401 or fewer refutes) and on the whole game against 175,425: supported at 97/100 of it or less, refuted at
99/100 or more once every rebuild finished."""
from __future__ import annotations

from harness.floor_g4 import SPEC as G4, _key, g4_library
from harness.floor_j1 import j1_bits, j1_piece, j1_valid
from harness.floor_u1 import u1_choose
from harness.floor_v2 import v2_assemble
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.floor_x2 import SPEC as X2, x2_library, x2_piece, x2_size
from harness.floor_z1 import z1_library
from harness.steady_state import map_bits

SPEC = {**{k: v for k, v in G4.items() if k != "taken"}, "measurement_fp": X2["measurement_fp"],
        "library_from": X2["library_from"], "builder": {**G4["builder"], "mirror": True}, "kept": X2["kept"],
        "z_held": X2["z_held"], "x2_held": [2, 4, 68, 71, 72, 118, 130], "control_bits": 175_425}

x3_report = w6_report


def x3_size(e: dict, x2: tuple, z: tuple, earlier: list, g4: tuple, g3: dict, assembly: dict, spec: dict) -> dict:
    """`x2`, `z` and `g4` are (run, its SPEC), `earlier` [(run, its SPEC), ...]; `assembly` holds J1's opening, below,
    base, map_cost and G3's library. `x2_size` passing on the registered bits is what proves the earlier runs."""
    x2_run, x2_spec = x2
    (z_run, _), (g4_run, g4_spec) = z, g4
    if (_integrity(e, spec)
            or x2_size(x2_run, z, earlier, g4, g3, assembly, x2_spec).get("bits") != spec["control_bits"]):
        return {"verdict": "not_run"}
    at = {r["index"]: _key(r) for r in g3["roots"]}
    if sorted(r["index"] for r in e["roots"]) != spec["indices"] or any(_key(r) != at[r["index"]] for r in e["roots"]):
        return {"verdict": "not_run"}
    rows = [r for run, _ in earlier for r in run["roots"]]
    before, _ = v2_assemble(rows, assembly)
    g4_lib = g4_library(assembly["library"], g3, [run for run, _ in earlier], g4_spec)
    library = z1_library(g4_lib, g4_run, spec)
    held = [r for r in x2_run["roots"] if r["index"] in spec["x2_held"]]
    if sorted(r["index"] for r in held) != spec["x2_held"] or not all(r["complete"] for r in held):
        return {"verdict": "not_run"}
    done = [r for r in e["roots"] if r["complete"]]
    if not all(j1_valid(r, x2_library(library)) for r in done):
        return {"verdict": "not_run"}
    level_bits = spec["builder"]["level_bits"]
    cost = {**assembly["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, level_bits)
                                       for i, (m, empty) in enumerate(library, 1) if i > len(assembly["library"])}}
    base = {**assembly["base"], **{_key(r): j1_piece(r, g4_lib) for r in g4_run["roots"] if r["index"] in spec["kept"]}}
    built = {**{k: j1_piece(rows[n], assembly["library"]) for k, n in before.items()},
             **{_key(r): j1_piece(r, library) for r in z_run["roots"] if r["index"] in spec["z_held"]},
             **{_key(r): x2_piece(r, library) for r in held}}
    rebuilds = {_key(r): x2_piece(r, library) for r in done}

    def size(chosen):
        return j1_bits(assembly["opening"], assembly["below"], {**base, **{k: rebuilds[k] for k in chosen}}, built,
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
