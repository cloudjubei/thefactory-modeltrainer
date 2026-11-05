"""§3.6 V1 — every winning move in batches, the whole game re-assembled. E3 found the builder's free pick the
cheapest winning move only where it had moved off the opening's (h348), and J1 (harness.floor_j1) assembles the whole
game from every build there is (203,774 bits from E1, O1 and E3). V1 is the first batch: at each ply-6 position with
more than one winning move under which G3 pays at least 1,500 bits (the scope, evidence/c49_o1_scope.json.gz), each
winning move no root of E1, O1 or E3 played there (a root left as a leaf played none) — 63 builds over 22 positions
— each with that move played at the root and otherwise as E1 built its 8 (the rule below the root, pipeline, G3's
library, 6-hour cap and budgets). Judged on completion out of 63 (58 supports, 49 or fewer refutes) and on the whole
game: J1's assembly of the earlier runs' builds with V1's complete ones added, against the same assembly without them
(203,774) — never worse than it, since that game stands, and a tie keeps the earlier build — supported at 97/100 of
it or less, refuted at 99/100 or more once every build finished."""
from __future__ import annotations

from harness.floor_e1 import SPEC as E1
from harness.floor_e3 import SPEC as E3, _root_move
from harness.floor_j1 import j1_choose, j1_piece, j1_valid
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report

SPEC = {**E1, "measurement_fp": E3["measurement_fp"], "scope": E3["scope"], "min_g3_bits": 1_500,
        "earlier": ["c49_e1.json.gz", "c49_o1.json.gz", "c49_e3.json.gz"], "roots": 63,
        "indices": [1, 3, 4, 13, 14, 41, 42, 71, 72, 73, 80, 81, 89, 99, 118, 124, 125, 126, 135, 140, 141, 143],
        "root_moves": [[1, 2], [1, 4], [3, 1], [4, 2], [4, 5], [13, 0], [13, 1], [13, 3], [13, 6], [14, 2], [14, 3],
                       [41, 1], [41, 4], [41, 5], [42, 1], [71, 0], [71, 2], [72, 0], [72, 5], [73, 1], [73, 3],
                       [73, 4], [73, 5], [80, 0], [80, 1], [80, 2], [80, 5], [80, 6], [81, 0], [89, 0], [89, 3],
                       [99, 1], [99, 3], [99, 4], [118, 1], [118, 3], [118, 4], [118, 5], [124, 0], [124, 4],
                       [124, 5], [124, 6], [125, 0], [125, 1], [125, 4], [125, 6], [126, 0], [126, 4], [126, 5],
                       [135, 1], [135, 3], [135, 5], [140, 0], [140, 1], [140, 2], [140, 5], [140, 6], [141, 3],
                       [141, 4], [143, 0], [143, 1], [143, 2], [143, 5]],
        "complete_support": 58, "complete_refute": 49, "support": (97, 100), "refute": (99, 100), "j1_bits": 203_774}

v1_report = w6_report


def v1_selected(scope: dict, earlier: list, spec: dict) -> list | None:
    if scope["positions_fp"] != spec["positions_fp"] or scope["library_from"] != spec["library_from"]:
        return None
    played: dict = {}
    for r in earlier:
        played.setdefault(r["index"], set()).add(_root_move(r))
    return [[p["index"], a] for p in sorted(scope["positions"], key=lambda p: p["index"])
            if len(p["winning"]) > 1 and p["g3_bits"] >= spec["min_g3_bits"]
            for a in p["winning"] if a not in played.get(p["index"], set())]


def v1_size(v: dict, earlier: list, scope: dict, assembly: dict, spec: dict) -> dict:
    """`earlier` is [(run, its SPEC), ...]; `assembly` holds J1's opening, below, base, map_cost and library."""
    if _integrity(v, spec) or any(_integrity(e, s) for e, s in earlier):
        return {"verdict": "not_run"}
    rows = [r for e, _ in earlier for r in e["roots"]]
    if v1_selected(scope, rows, spec) != spec["root_moves"]:
        return {"verdict": "not_run"}
    if (sorted([r["index"], r["root_move"]] for r in v["roots"]) != sorted(spec["root_moves"])
            or any(_root_move(r) != r["root_move"] for r in v["roots"])):
        return {"verdict": "not_run"}
    library = assembly["library"]
    if not all(j1_valid(r, library) for r in rows + [r for r in v["roots"] if r["complete"]]):
        return {"verdict": "not_run"}

    def choose(pieces):
        candidates: dict = {}
        for n, r in pieces:
            candidates.setdefault((tuple(r["board"]), r["to_move"]), []).append(
                ((n, r["index"], _root_move(r)), j1_piece(r, library)))
        return j1_choose(assembly["opening"], assembly["below"], assembly["base"], candidates, assembly["map_cost"])

    earlier_pieces = [(0, r) for r in rows]
    _, control = choose(earlier_pieces)
    if control != spec["j1_bits"]:
        return {"verdict": "not_run"}
    chosen, bits = choose(earlier_pieces + [(1, r) for r in v["roots"] if r["complete"]])
    taken = sorted([i, a] for n, i, a in chosen.values() if n == 1)
    if bits > control:
        bits, taken = control, []
    unfinished = not all(r["complete"] for r in v["roots"])
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if not unfinished and rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control, "taken": taken,
            "unfinished": unfinished}
