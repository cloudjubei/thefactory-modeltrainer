"""§3.6 X4 — the control X3 did not have. X3 rebuilt the 409 ply-8 strategies with both readings and the whole game
fell to 166,623 bits, but the pieces those rebuilds stand in for came from smaller libraries, and the 391 rebuilds that
played no mirrored leaf clear the bar alone (h370), so X3 cannot say what the mirror itself is worth. X4 rebuilds the 18
that played one, at the same positions with G4's builder, rule, pipeline and cap, from the same list of 5,196 maps as X3
(its `library_from`) with `mirror` off: the first half of the list is then read plain like the second, so every
reference costs what it cost in X3 and the second reading is the only change to the builder. The whole game is X3's,
recomputed by `x3_size`, with the 18 in X3's place and U1's search run again; the 391 stay as X3 built them, taken
to be what the mirror off would have built, since a mirrored reading they did not keep leaves their strategies alone.
Judged on completion out of 18 (all supports, 16 or fewer refutes) and on what the second reading is worth: supported
when the game takes 101/100 of 166,623 or more with the mirror off, refuted at 401/400 or less. An unfinished rebuild
falls back to the bigger piece it stands in for, so it can refute and cannot support."""
from __future__ import annotations

from harness.floor_g4 import _key
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.floor_x3 import SPEC as X3, x3_size

SPEC = {**X3, "roots": 18,
        "indices": [23, 55, 57, 63, 86, 114, 116, 182, 198, 200, 275, 280, 335, 344, 499, 554, 569, 594],
        "builder": {k: v for k, v in X3["builder"].items() if k != "mirror"}, "complete_support": 18,
        "complete_refute": 16, "support": (101, 100), "refute": (401, 400), "control_bits": 166_623}

x4_report = w6_report


def _mirrored(r: dict) -> bool:
    return any(1 <= n.get("leaf", 0) <= r["shared"] // 2 for _, _, n in r["nodes"])


def x4_size(e: dict, x3: tuple, x2: tuple, z: tuple, earlier: list, g4: tuple, g3: dict, assembly: dict,
            spec: dict) -> dict:
    """`x3` is (X3's run, its SPEC) and the rest what `x3_size` takes; `x3_size` giving the registered bits is what
    proves the earlier runs."""
    x3_run, x3_spec = x3
    rows = {r["index"]: r for r in e["roots"]}
    at = {r["index"]: _key(r) for r in x3_run["roots"]}
    if (_integrity(e, spec) or sorted(rows) != spec["indices"] or any(_key(r) != at.get(i) for i, r in rows.items())
            or any(_mirrored(r) for r in e["roots"] if r["complete"])):
        return {"verdict": "not_run"}
    judged = (x2, z, earlier, g4, g3, assembly, x3_spec)
    control = x3_size(x3_run, *judged).get("bits")
    off = x3_size({**x3_run, "roots": [rows.get(r["index"], r) for r in x3_run["roots"]]}, *judged)
    if control != spec["control_bits"] or off["verdict"] == "not_run":
        return {"verdict": "not_run"}
    bits, unfinished = off["bits"], off["unfinished"]
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("refuted" if rden * bits <= rnum * control else
               "supported" if not unfinished and den * bits >= num * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control,
            "rebuilt": off["rebuilt"], "unfinished": unfinished}
