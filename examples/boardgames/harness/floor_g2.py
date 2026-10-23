"""§3.6 G2 — the whole game, a second pass. G1 built F2's 16 heaviest positions again from every map their control
had found, all within reach, and took three quarters of the control's bits, smaller on 15 of 16 (h314, h319). G2 is
F2 (harness.floor_f2) built once more the same way: every one of the 671 positions with L5's rule and pipeline, no
waves, each build starting from all of F2's found maps in its order (the run's `library_from`), the window raised past
any build's own maps — only that changed, no measured code. Judged on completion as F1, and on the whole game's size
— the opening, every position's own bits with references coded by first_use (harness.reference_coding, h312) and
every library map some leaf plays paid once — against F2's coded the same way (398,195 bits, h312: the opening, L2's
seed library and F2's positions): supported at nine tenths of it or less, refuted at F2's or more. An unfinished root
counts without its references, a lower bound under any code."""
from __future__ import annotations

from harness.floor_f1 import f1_checked, f1_complete
from harness.floor_f2 import SPEC as F2
from harness.floor_g1 import _coded, g1_library
from harness.floor_k1 import SPEC as K1
from harness.floor_l5 import SPEC as L5
from harness.steady_state import map_bits

SPEC = {**{k: v for k, v in F2.items() if k not in ("f1_bits", "wave_size", "library_bits")},
        "measurement_fp": K1["measurement_fp"], "library_from": "c49_f2.json.gz",
        "builder": {**L5["builder"], "reuse_window": 100_000}, "f2_bits": 398_195}

g2_complete = f1_complete
g2_library = g1_library


def _opening(e: dict) -> int:
    return e["opening"]["moves"] * 4 + e["opening"]["trivial"]


def g2_size(e: dict, f2: dict, spec: dict) -> dict:
    if not f1_checked(e, spec) or not f1_checked(f2, F2) or not all(r["complete"] for r in f2["roots"]):
        return {"verdict": "not_run"}
    control = _opening(f2) + F2["library_bits"] + sum(_coded(r) for r in f2["roots"])
    if control != spec["f2_bits"]:
        return {"verdict": "not_run"}
    library = g2_library(f2)
    for r in e["roots"]:
        if (r["shared"] != len(library) or r["maps"][1:1 + len(library)] != [m for m, _ in library]
                or r["map_empty"][1:1 + len(library)] != [empty for _, empty in library]):
            return {"verdict": "not_run"}
    played = {n["leaf"] for r in e["roots"] for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}
    shared = sum(map_bits({int(c): k for c, k in library[i - 1][0].items()}, library[i - 1][1],
                          spec["builder"]["level_bits"]) for i in played)
    bits = _opening(e) + sum(_coded(r) for r in e["roots"]) + shared
    lower = not all(r["complete"] for r in e["roots"])
    num, den = spec["support"]
    verdict = ("refuted" if bits >= control else
               "supported" if not lower and den * bits <= num * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control, "lower_bound": lower,
            "library_maps": len(played)}
