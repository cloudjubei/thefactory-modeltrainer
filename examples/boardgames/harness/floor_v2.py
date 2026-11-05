"""§3.6 V2 — every winning move by transposition group. V1 added 63 builds and the whole game took 8 (h351, h352): a
build at a ply-6 position whose ply-8 positions another ply-6 position shares pays only when that position swaps
too, and J2 (harness.floor_j2), choosing by group, found nothing more because the positions sharing them had no
builds — six of them with a single winning move, which no batch of every winning move at the positions with a choice
builds (h353). V2 picks by group: the whole game is assembled from E1's, O1's, E3's and V1's builds (J1's chooser,
then J2 from its choice: 197,650 bits), the ply-6 positions are joined into groups by the ply-8 positions they share,
and each group is weighed by what the game pays there now — the opening's move and the trivial positions and G3's
strategies below each member keeping G3's piece (each ply-8 position once), and the builds taken at the others,
library maps aside. In every group paying at least 4,000 bits, each winning move no earlier root played at a member
with more than one, and at a member with one winning move and no earlier build a free build (no root move: the
builder may also leave the root a leaf) — 67 builds over 9 groups, otherwise as E1 built its 8 (rule, pipeline, G3's
library, 6-hour cap, budgets). Judged on completion out of 67 (62 supports, 52 or fewer refutes) and on the whole
game assembled the same way with V2's complete builds added, against 197,650 — never worse than it, a tie keeping
the earlier build — supported at 97/100 of it or less, refuted at 99/100 or more once every build finished."""
from __future__ import annotations

from harness.floor_e1 import SPEC as E1
from harness.floor_e3 import SPEC as E3, _root_move
from harness.floor_j1 import j1_choose, j1_piece, j1_valid
from harness.floor_j2 import j2_choose, j2_groups
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report

SPEC = {**E1, "measurement_fp": E3["measurement_fp"], "scope": E3["scope"], "min_group_bits": 4_000,
        "earlier": ["c49_e1.json.gz", "c49_o1.json.gz", "c49_e3.json.gz", "c49_v1.json.gz"], "roots": 67,
        "indices": [0, 2, 44, 47, 50, 54, 58, 67, 68, 70, 82, 83, 84, 86, 87, 88, 91, 96, 98, 104, 105, 110,
                    111, 112, 114, 137],
        "root_moves": [[0, 2], [0, 4], [2, None], [44, 1], [44, 3], [47, 1], [47, 3], [50, 1], [50, 3], [50, 4],
                       [54, 1], [54, 3], [58, None], [67, 0], [67, 3], [68, None], [70, 0], [70, 3], [82, None],
                       [83, 1], [83, 3], [84, None], [86, 1], [86, 2], [86, 3], [86, 4], [87, 1], [87, 2], [87, 3],
                       [87, 4], [87, 6], [88, 1], [88, 2], [91, 0], [91, 3], [91, 6], [96, None], [98, 0], [98, 1],
                       [98, 3], [98, 5], [104, 0], [104, 1], [104, 2], [104, 3], [104, 6], [105, 0], [105, 1],
                       [105, 4], [110, 0], [110, 3], [111, 0], [111, 1], [111, 3], [111, 4], [111, 5], [112, 0],
                       [112, 3], [114, 0], [114, 1], [114, 2], [114, 3], [114, 4], [114, 6], [137, 1], [137, 3],
                       [137, 4]],
        "complete_support": 62, "complete_refute": 52, "support": (97, 100), "refute": (99, 100),
        "control_bits": 197_650}

v2_report = w6_report


def _order(move: list) -> tuple:
    return move[0], -1 if move[1] is None else move[1]


def v2_assemble(rows: list, assembly: dict) -> tuple:
    """The whole game from `rows` (builds): J1's chooser, then J2 from its choice; returns ({position: the index in
    `rows` of the build it takes}, bits)."""
    candidates: dict = {}
    for n, r in enumerate(rows):
        candidates.setdefault((tuple(r["board"]), r["to_move"]), []).append((n, j1_piece(r, assembly["library"])))
    args = (assembly["opening"], assembly["below"], assembly["base"], candidates, assembly["map_cost"])
    start, _ = j1_choose(*args)
    return j2_choose(*args, start)


def v2_selected(scope: dict, earlier: list, assembly: dict, spec: dict) -> list | None:
    if scope["positions_fp"] != spec["positions_fp"] or scope["library_from"] != spec["library_from"]:
        return None
    chosen, _ = v2_assemble(earlier, assembly)
    below, base, library = assembly["below"], assembly["base"], assembly["library"]
    index = {k: i for i, k in enumerate(sorted(below))}
    winning = {p["index"]: p["winning"] for p in scope["positions"]}
    played: dict = {}
    for r in earlier:
        played.setdefault(r["index"], set()).add(_root_move(r))
    selected = []
    for group in j2_groups(below):
        kept = [k for k in group if k not in chosen]
        kids = {c for k in kept for c in below[k]}
        paid = (4 * len(kept) + sum(base[c][0] if c in base else 1 for c in kids)
                + sum(j1_piece(earlier[chosen[k]], library)[0] for k in group if k in chosen))
        if paid < spec["min_group_bits"]:
            continue
        for k in group:
            i = index[k]
            if len(winning[i]) > 1:
                selected += [[i, a] for a in winning[i] if a not in played.get(i, set())]
            elif i not in played:
                selected.append([i, None])
    return sorted(selected, key=_order)


def v2_size(v: dict, earlier: list, scope: dict, assembly: dict, spec: dict) -> dict:
    """`earlier` is [(run, its SPEC), ...]; `assembly` holds J1's opening, below, base, map_cost and library."""
    if _integrity(v, spec) or any(_integrity(e, s) for e, s in earlier):
        return {"verdict": "not_run"}
    rows = [r for e, _ in earlier for r in e["roots"]]
    index = {k: i for i, k in enumerate(sorted(assembly["below"]))}
    if any(index.get((tuple(r["board"]), r["to_move"])) != r["index"] for r in rows + v["roots"]):
        return {"verdict": "not_run"}
    if v2_selected(scope, rows, assembly, spec) != spec["root_moves"]:
        return {"verdict": "not_run"}
    if (sorted(([r["index"], r["root_move"]] for r in v["roots"]), key=_order) != spec["root_moves"]
            or any(r["root_move"] is not None and _root_move(r) != r["root_move"] for r in v["roots"])):
        return {"verdict": "not_run"}
    library = assembly["library"]
    if not all(j1_valid(r, library) for r in rows + [r for r in v["roots"] if r["complete"]]):
        return {"verdict": "not_run"}
    _, control = v2_assemble(rows, assembly)
    if control != spec["control_bits"]:
        return {"verdict": "not_run"}
    pieces = rows + [r for r in v["roots"] if r["complete"]]
    chosen, bits = v2_assemble(pieces, assembly)
    taken = sorted(([pieces[n]["index"], pieces[n]["root_move"]] for n in chosen.values() if n >= len(rows)),
                   key=_order)
    if bits > control:
        bits, taken = control, []
    unfinished = not all(r["complete"] for r in v["roots"])
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if not unfinished and rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control, "taken": taken,
            "unfinished": unfinished}
