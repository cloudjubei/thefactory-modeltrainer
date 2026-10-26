"""§3.6 G3 — the whole game, a third pass. G2 (harness.floor_g2) built F2's 671 positions again from every map F2
found, all within reach, and took 293,843 bits against F2's 398,195 (0.74, h321), smaller on 540 positions (h322).
G3 is G2 once more: every position with L5's rule and pipeline, no waves, each build starting from every map G2's
whole game holds — the library maps its leaves play, in F2's order, then the maps its own builds found, in root
order (`played_library`) — written by scripts/c4_library_file.py from LIBRARY_SOURCES to the run's `library_from`.
Only that changed, no measured code. Judged on completion as F1, and on the whole game's size coded as G2's —
the opening, every position's own bits with references by first_use and every library map some leaf plays paid
once — against G2's, recomputed from its evidence and F2's: supported at nine tenths of it or less, refuted at
G2's or more. An unfinished root counts without its references, a lower bound under any code."""
from __future__ import annotations

from harness.floor_f1 import f1_checked, f1_complete
from harness.floor_g1 import _coded, g1_library
from harness.floor_g2 import SPEC as G2, g2_size
from harness.steady_state import map_bits

SPEC = {**G2, "library_from": "c49_g3_library.json.gz", "g2_bits": 293_843}
LIBRARY_SOURCES = ("c49_g2.json.gz", "c49_f2.json.gz")

g3_complete = f1_complete


def played_library(e: dict, library: list) -> list:
    played = sorted({n["leaf"] for r in e["roots"] for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)})
    return [library[i - 1] for i in played] + [m for r in e["roots"] for m in r["found_maps"]]


def g3_library(g2: dict, f2: dict) -> list:
    return played_library(g2, g1_library(f2))


def g3_size(e: dict, g2: dict, f2: dict, spec: dict) -> dict:
    control = g2_size(g2, f2, {**G2, "f2_bits": spec["f2_bits"]})
    if (not f1_checked(e, spec) or control["verdict"] == "not_run" or control["lower_bound"]
            or control["bits"] != spec["g2_bits"]):
        return {"verdict": "not_run"}
    library = g3_library(g2, f2)
    for r in e["roots"]:
        if (r["shared"] != len(library) or r["maps"][1:1 + len(library)] != [m for m, _ in library]
                or r["map_empty"][1:1 + len(library)] != [empty for _, empty in library]):
            return {"verdict": "not_run"}
    played = {n["leaf"] for r in e["roots"] for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}
    shared = sum(map_bits({int(c): k for c, k in library[i - 1][0].items()}, library[i - 1][1],
                          spec["builder"]["level_bits"]) for i in played)
    bits = e["opening"]["moves"] * 4 + e["opening"]["trivial"] + sum(_coded(r) for r in e["roots"]) + shared
    lower = not all(r["complete"] for r in e["roots"])
    num, den = spec["support"]
    verdict = ("refuted" if bits >= control["bits"] else
               "supported" if not lower and den * bits <= num * control["bits"] else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control["bits"], "ratio": bits / control["bits"],
            "lower_bound": lower, "library_maps": len(played)}
