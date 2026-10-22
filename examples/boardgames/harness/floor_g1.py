"""§3.6 G1 — maps shared across the heavy tail. A build tries the maps it already holds most recent first, only the
last `reuse_window` (200) of them, and a library sits at the front of that list: F2's heavy builds started from
800-2,500 library maps and 151 of their 2,208 leaves played one (h275: 880 of F1's 8,904). G1 builds B1's positions
(F2's 16 heaviest) with L5's rule and pipeline, starting from every map B1's control found on them (766, the run's
`library_from`, harness.floor_w3), the window raised past any build's own maps (the control's builds kept at most 116)
so that every library map is tried — only that changed, no measured code. Judged on completion out of 16 (14 supports,
12 or fewer refutes) and on its total against the control's (harness.floor_b1_control, judged under its own
registration) with both arms' references coded by first_use (harness.reference_coding, h312) and every library map
some leaf plays paid once: supported at nine tenths of it or less, refuted at the control's or more. An unfinished root
counts without its references, a lower bound under any code."""
from __future__ import annotations

import math

from games.connect4 import C4State, Connect4
from harness.floor_b1 import SPEC as B1
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_k1 import SPEC as K1
from harness.floor_l5 import SPEC as L5
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.reference_coding import code_bits, walk_leaves
from harness.steady_state import map_bits

SPEC = {**B1, "measurement_fp": K1["measurement_fp"], "library_from": "c49_b1_control.json.gz",
        "builder": {**L5["builder"], "reuse_window": 100_000}}

g1_report = w6_report


def g1_library(control: dict) -> list:
    return [(levels, empty) for r in control["roots"] for levels, empty in r["found_maps"]]


def _coded(r: dict) -> int:
    n_maps, shared = len(r["maps"]), r["shared"]
    if not r["complete"]:
        return r["bits"]["nodes"] - r["bits"]["leaves"] * (math.ceil(math.log2(n_maps)) if n_maps > 1 else 0)
    game = Connect4()
    nodes = {(tuple(b), t): n for b, t, n in r["nodes"]}
    refs = walk_leaves(game, C4State(tuple(r["board"]), r["to_move"], None, False), nodes)
    return r["bits"]["nodes"] - code_bits("fixed", refs, n_maps, shared) + code_bits("first_use", refs, n_maps, shared)


def g1_size(e: dict, control: dict) -> dict:
    if _integrity(e, SPEC) or _integrity(control, CONTROL) or not all(r["complete"] for r in control["roots"]):
        return {"verdict": "not_run"}
    library = g1_library(control)
    for r in e["roots"]:
        if (r["shared"] != len(library) or r["maps"][1:1 + len(library)] != [m for m, _ in library]
                or r["map_empty"][1:1 + len(library)] != [empty for _, empty in library]):
            return {"verdict": "not_run"}
    played = {n["leaf"] for r in e["roots"] for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}
    shared = sum(map_bits({int(c): k for c, k in library[i - 1][0].items()}, library[i - 1][1],
                          SPEC["builder"]["level_bits"]) for i in played)
    bits = sum(_coded(r) for r in e["roots"]) + shared
    control_bits = sum(_coded(r) for r in control["roots"])
    ratio = bits / control_bits
    lower = not all(r["complete"] for r in e["roots"])
    verdict = "refuted" if ratio >= 1 else "supported" if not lower and ratio <= SPEC["support"] else "inconclusive"
    return {"verdict": verdict, "bits": bits, "control_bits": control_bits, "ratio": ratio, "lower_bound": lower,
            "library_maps": len(played)}
