"""§3.6 E3 — every winning move at ply 6, against the builder's pick. E1 and O1 saved most where the builder's free
pick (the winning move whose replies the most known maps cover) replaced the opening's: does building under every
winning move and keeping the cheapest beat that pick? E3 takes E1's 8 heaviest ply-6 positions, whose picks E1
already built, and builds each other winning move there (the scope's winning moves, evidence/c49_o1_scope.json.gz,
less the move E1's root played): 13 builds over 7 positions (#39 has one winning move), each with that move played at
the root (the builder's root_move) and otherwise as E1 built its 8 — the rule below the root, pipeline, G3's library,
6-hour cap and budgets. Judged on completion out of 13 (12 supports, 10 or fewer refutes) and on the bits of the
cheapest complete build at each position against E1's builds for the same 7 — each charged its strategy by first_use
plus the library maps it plays that G3's whole game does not (a tie keeps E1's) — supported at nine tenths of E1's
bits or less, refuted at 97/100 or more once every build finished (the saving under 3%)."""
from __future__ import annotations

from harness.floor_e1 import SPEC as E1
from harness.floor_g1 import _coded
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.steady_state import map_bits

SPEC = {**E1, "measurement_fp": "3981ea9f5020", "roots": 13, "indices": [18, 69, 79, 115, 130, 145, 146],
        "root_moves": [[18, 4], [69, 1], [69, 6], [79, 1], [79, 4], [79, 6], [115, 2], [130, 2], [145, 0], [145, 1],
                       [145, 4], [146, 4], [146, 5]],
        "scope": "c49_o1_scope.json.gz", "complete_support": 12, "complete_refute": 10, "support": (9, 10),
        "refute": (97, 100), "e1_bits": 28_712}

e3_report = w6_report


def _root_move(r: dict):
    key = (tuple(r["board"]), r["to_move"])
    return next((n.get("move") for b, m, n in r["nodes"] if (tuple(b), m) == key), None)


def e3_selected(scope: dict, e1: dict, spec: dict) -> list | None:
    if scope["positions_fp"] != spec["positions_fp"] or scope["library_from"] != spec["library_from"]:
        return None
    winning = {p["index"]: p["winning"] for p in scope["positions"]}
    played = {r["index"]: _root_move(r) for r in e1["roots"]}
    return [[i, a] for i in sorted(played) for a in winning[i] if a != played[i]]


def e3_size(e: dict, e1: dict, g3: dict, library: list, scope: dict, spec: dict) -> dict:
    if _integrity(e, spec) or _integrity(e1, E1) or not all(r["complete"] for r in e1["roots"]):
        return {"verdict": "not_run"}
    if e3_selected(scope, e1, spec) != spec["root_moves"]:
        return {"verdict": "not_run"}
    if (sorted([r["index"], r["root_move"]] for r in e["roots"]) != sorted(spec["root_moves"])
            or any(_root_move(r) != r["root_move"] for r in e["roots"])):
        return {"verdict": "not_run"}
    for r in e["roots"] + e1["roots"]:
        if (r["shared"] != len(library) or r["maps"][1:1 + len(library)] != [m for m, _ in library]
                or r["map_empty"][1:1 + len(library)] != [empty for _, empty in library]):
            return {"verdict": "not_run"}

    def played(rs):
        return {n["leaf"] for r in rs for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}

    g3_played = played(g3["roots"])

    def cost(rs):
        return sum(_coded(r) for r in rs) + sum(
            map_bits({int(c): k for c, k in library[i - 1][0].items()}, library[i - 1][1],
                     spec["builder"]["level_bits"]) for i in played(rs) - g3_played)

    picks = {r["index"]: r for r in e1["roots"] if r["index"] in spec["indices"]}
    control = cost(list(picks.values()))
    if control != spec["e1_bits"]:
        return {"verdict": "not_run"}
    best = [min([picks[i], *(r for r in e["roots"] if r["index"] == i and r["complete"])], key=lambda r: cost([r]))
            for i in spec["indices"]]
    bits = cost(best)
    unfinished = not all(r["complete"] for r in e["roots"])
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if not unfinished and rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control,
            "moves": {i: _root_move(r) for i, r in zip(spec["indices"], best)}, "unfinished": unfinished}
