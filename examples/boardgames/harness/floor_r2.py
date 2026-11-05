"""§3.6 R2 — the builder counting references as the judges do. Since R1 (h312) every judge codes a leaf's reference
into the map list by first_use (harness.reference_coding: the library maps played named once, then move-to-front
ranks in walk order), but the builder still prices each reference at the index's full width when it decides by size
— 10 bits a leaf against G1's 767-map lists, where a repeated map costs the judges 1-3 — so improvement may undo
splits that would pay. R2 is G1 (harness.floor_g1: B1's 16 heaviest positions from the 766 maps B1's control found,
all within reach) with only the builder's "refs" set to "first_use" (harness.strategy_builder). Judged on completion
out of 16 (14 supports, 12 or fewer refutes) and on its total against G1's 74,556 bits (h314), recomputed from G1's
evidence and B1's control, both coded by first_use with every library map some leaf plays paid once: supported at nine
tenths of it or less, refuted at G1's or more. R2's builds already count their references by first_use, so the judge
takes off what the builder charged (harness.strategy_builder.first_use_references) and adds the judges' code — the
same on a finished build, whose own maps are all played — where harness.floor_g1._coded would recode a fixed-width
count; an unfinished root counts without its references, a lower bound."""
from __future__ import annotations

from games.connect4 import C4State, Connect4
from harness.floor_g1 import SPEC as G1, g1_library, g1_size
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.reference_coding import code_bits, walk_leaves
from harness.steady_state import map_bits
from harness.strategy_builder import first_use_references

SPEC = {**G1, "measurement_fp": "77fc4515e75e", "builder": {**G1["builder"], "refs": "first_use"}, "g1_bits": 74_556}

r2_report = w6_report


def r2_coded(r: dict) -> int:
    game = Connect4()
    root = C4State(tuple(r["board"]), r["to_move"], None, False)
    nodes = {(tuple(b), t): n for b, t, n in r["nodes"]}
    bits = r["bits"]["nodes"] - first_use_references(game, root, nodes, r["shared"])
    if not r["complete"]:
        return bits
    return bits + code_bits("first_use", walk_leaves(game, root, nodes), len(r["maps"]), r["shared"])


def r2_size(e: dict, g1: dict, control: dict, spec: dict) -> dict:
    before = g1_size(g1, control)
    if (_integrity(e, spec) or before["verdict"] == "not_run" or before["lower_bound"]
            or before["bits"] != spec["g1_bits"]):
        return {"verdict": "not_run"}
    library = g1_library(control)
    for r in e["roots"]:
        if (r["shared"] != len(library) or r["maps"][1:1 + len(library)] != [m for m, _ in library]
                or r["map_empty"][1:1 + len(library)] != [empty for _, empty in library]):
            return {"verdict": "not_run"}
    played = {n["leaf"] for r in e["roots"] for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}
    shared = sum(map_bits({int(c): k for c, k in library[i - 1][0].items()}, library[i - 1][1],
                          spec["builder"]["level_bits"]) for i in played)
    bits = sum(r2_coded(r) for r in e["roots"]) + shared
    lower = not all(r["complete"] for r in e["roots"])
    ratio = bits / before["bits"]
    verdict = "refuted" if ratio >= 1 else "supported" if not lower and ratio <= spec["support"] else "inconclusive"
    return {"verdict": verdict, "bits": bits, "control_bits": before["bits"], "ratio": ratio, "lower_bound": lower,
            "library_maps": len(played)}
