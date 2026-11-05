"""§3.6 O1 — the builder's move at ply 6, across the opening. E1's saving was that move (h342): the opening picks by
the replies the empty map wins, the builder by the replies any known map covers. scripts/c4_library_pick_scope.py
(evidence/c49_o1_scope.json.gz) finds 34 of the opening's 148 ply-6 positions where the builder's rule, given G3's
library, picks another move; O1 takes the 31 beyond E1's 8 — 51,312 of G3's bits below them — and builds each from
ply 6 exactly as E1 built its 8 (the same rule, pipeline, library, 6-hour cap and budgets; no measured code changed).
Judged as E1 (harness.floor_e1.e1_size) once the scope evidence names the same 31: completion out of 31 (29 supports,
25 or fewer refutes) and the bits against G3's for the same subtrees, supported at three quarters of them or less,
refuted at G3's or more."""
from __future__ import annotations

from harness.floor_e1 import SPEC as E1, e1_size
from harness.floor_w6 import w6_report

SPEC = {**E1, "roots": 31, "excluded": E1["indices"], "scope": "c49_o1_scope.json.gz",
        "indices": [3, 13, 14, 21, 28, 37, 41, 42, 43, 46, 48, 52, 54, 73, 77, 78, 80, 81, 89, 92, 97, 103, 105, 113,
                    118, 124, 125, 126, 127, 135, 140],
        "complete_support": 29, "complete_refute": 25, "support": (3, 4), "g3_subtree_bits": 51_312}

o1_report = w6_report


def o1_selected(scope: dict, spec: dict) -> list | None:
    if scope["positions_fp"] != spec["positions_fp"] or scope["library_from"] != spec["library_from"]:
        return None
    return sorted(p["index"] for p in scope["positions"]
                  if p["library_pick"] != p["opening"] and p["index"] not in spec["excluded"])


def o1_size(e: dict, g3: dict, library: list, moves: dict, scope: dict, spec: dict) -> dict:
    if o1_selected(scope, spec) != spec["indices"]:
        return {"verdict": "not_run"}
    return e1_size(e, g3, library, moves, spec)
