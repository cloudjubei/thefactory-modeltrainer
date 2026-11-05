"""§3.6 E1 — earlier leaves. WeakC4's edge is a small tree of large leaves reached early (h334): its leaves start at
ply 4, ours at ply 8, under an opening fixed to ply 8 before any map was found. E1 takes the 8 ply-6 positions of the
opening (harness.opening at ply 6: 148 positions the empty map does not win, sorted as scripts/c4_strategy_s2.py
sorts them) under which G3 paid most, and builds each from ply 6 with G3's rule, pipeline and library — so a leaf may
sit at ply 6 and the move there is the builder's — with 6 hours a position (each holds up to 7 of G3's heavy
positions) and 20-minute searches at ply 6 and ply 8, 30 s below. Only the root's ply and the budgets changed, no
measured code. Judged on completion out of 8 (7 supports, 5 or fewer refutes) and on the bits against G3's for the
same subtrees — the opening's move at the position (4 bits), the positions below it the empty map wins (1 bit each)
and G3's strategies for the rest, references by first_use, 47,038 bits — the arm's strategies by first_use plus every
library map they play that G3's whole game does not: supported at four fifths of it or less, refuted at G3's or more.
An unfinished root counts without its references, a lower bound."""
from __future__ import annotations

from games.connect4 import C4State, Connect4
from harness.floor_g1 import _coded
from harness.floor_g3 import SPEC as G3
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.steady_state import map_bits

SPEC = {"positions_fp": G3["positions_fp"], "measurement_fp": "77fc4515e75e", "source": "opening", "ply": 6,
        "seed": G3["seed"], "frontier": 148, "roots": 8, "indices": [18, 39, 69, 79, 115, 130, 145, 146],
        "complete_support": 7, "complete_refute": 5, "library_from": G3["library_from"],
        "builder": {**G3["builder"], "seconds": 21_600.0, "budgets": [[0, 1200.0], [2, 1200.0], [4, 30.0]]},
        "search": G3["search"], "support": (4, 5), "g3_subtree_bits": 47_038}

e1_report = w6_report

GAME = Connect4()


def _below(state, roots: dict, moves: dict) -> tuple:
    child = GAME.step(state, moves[GAME.state_key(state)])
    kids = [GAME.state_key(c) for c in (GAME.step(child, a) for a in GAME.legal_actions(child))
            if not GAME.is_terminal(c)]
    return kids, [roots[k] for k in kids if k in roots]


def e1_control(states: list, g3: dict, moves: dict) -> list:
    roots = {(tuple(r["board"]), r["to_move"]): r for r in g3["roots"]}
    out = []
    for s in states:
        kids, built = _below(s, roots, moves)
        out.append(4 + len(kids) - len(built) + sum(_coded(r) for r in built))
    return out


def e1_size(e: dict, g3: dict, library: list, moves: dict, spec: dict) -> dict:
    if _integrity(e, spec):
        return {"verdict": "not_run"}
    states = [C4State(tuple(r["board"]), r["to_move"], None, False) for r in e["roots"]]
    roots = {(tuple(r["board"]), r["to_move"]): r for r in g3["roots"]}
    if not all(r["complete"] for s in states for r in _below(s, roots, moves)[1]):
        return {"verdict": "not_run"}
    control = sum(e1_control(states, g3, moves))
    if control != spec["g3_subtree_bits"]:
        return {"verdict": "not_run"}
    for r in e["roots"]:
        if (r["shared"] != len(library) or r["maps"][1:1 + len(library)] != [m for m, _ in library]
                or r["map_empty"][1:1 + len(library)] != [empty for _, empty in library]):
            return {"verdict": "not_run"}

    def played(rs):
        return {n["leaf"] for r in rs for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}

    extra = sum(map_bits({int(c): k for c, k in library[i - 1][0].items()}, library[i - 1][1],
                         spec["builder"]["level_bits"]) for i in played(e["roots"]) - played(g3["roots"]))
    bits = sum(_coded(r) for r in e["roots"]) + extra
    lower = not all(r["complete"] for r in e["roots"])
    num, den = spec["support"]
    verdict = ("refuted" if bits >= control else
               "supported" if not lower and den * bits <= num * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control, "lower_bound": lower}
