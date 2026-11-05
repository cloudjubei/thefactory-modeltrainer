"""§3.6 G4 — the assembled whole game's library pass. G2 and G3 rebuilt every ply-8 position from every map the last
whole game held (0.74, then 0.955; h321, h325). Since then the whole game is assembled (harness.floor_j1, floor_j2,
floor_v2): 51 ply-6 positions take a strategy built from ply 6, and the rest keep G3's — 409 of G3's ply-8 strategies,
192,591 bits in all (h356). Each of those pieces pays its own maps inline, so a map one of them found is paid again
by any other piece that would play it. G4 rebuilds the 409 ply-8 positions the assembled game still takes from G3,
with G3's builder, rule and pipeline, from every map that game holds: G3's library in its order, then the own maps of
those 409 G3 strategies in position order, then the own maps of the 51 builds it takes, in the order of the runs and
positions (`g4_library`, written by scripts/c4_library_file.py to `library_from`). The ply-6 builds are not rebuilt.
The whole game is re-assembled with every complete rebuild in place of G3's strategy, a rebuild that alone grows the
game going back to G3's, then the ply-6 builds chosen again as floor_v2 chooses them — never worse than 192,591.
Judged on completion out of 409 (all supports, 401 or fewer refutes) and on the whole game against 192,591:
supported at 97/100 of it or less, refuted at 99/100 or more once every rebuild finished."""
from __future__ import annotations

from harness.floor_e3 import SPEC as E3
from harness.floor_g3 import SPEC as G3, g3_library
from harness.floor_j1 import j1_bits, j1_piece, j1_valid
from harness.floor_v2 import v2_assemble
from harness.floor_w1 import _integrity
from harness.floor_w6 import w6_report
from harness.steady_state import map_bits

SPEC = {**{k: v for k, v in G3.items() if k not in ("indices", "g2_bits", "f2_bits", "w9_bits", "w9_high")},
        "measurement_fp": E3["measurement_fp"], "library_from": "c49_g4_library.json.gz",
        "earlier": ["c49_e1.json.gz", "c49_o1.json.gz", "c49_e3.json.gz", "c49_v1.json.gz", "c49_v2.json.gz"],
        "roots": 409,
        "indices": [
                    13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 36, 37, 38, 40, 41, 43, 44,
                    46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 63, 64, 65, 66, 67, 68, 70, 71, 73, 74, 75, 76, 77, 83,
                    84, 85, 86, 87, 88, 89, 91, 94, 95, 97, 98, 99, 100, 103, 104, 105, 107, 108, 109, 110, 111, 112,
                    113, 114, 115, 116, 119, 120, 122, 123, 125, 126, 128, 129, 130, 131, 133, 134, 135, 136, 138, 139,
                    142, 143, 145, 146, 148, 149, 151, 153, 154, 157, 159, 160, 162, 163, 165, 166, 168, 170, 171, 172,
                    173, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 193, 194, 195, 196,
                    197, 198, 200, 201, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 233, 235, 237, 238, 239, 244,
                    245, 246, 254, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 272, 273, 274, 275, 276, 277,
                    278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 294, 300, 301, 302, 303, 304, 305, 306, 307, 309,
                    310, 311, 312, 324, 325, 326, 327, 329, 330, 331, 332, 333, 334, 335, 336, 337, 341, 342, 344, 345,
                    346, 348, 355, 356, 357, 358, 359, 360, 361, 362, 363, 367, 368, 369, 371, 372, 373, 376, 377, 378,
                    379, 380, 381, 382, 383, 384, 385, 386, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,
                    405, 406, 407, 408, 409, 410, 411, 412, 413, 415, 416, 424, 425, 426, 427, 428, 429, 436, 438, 439,
                    440, 441, 442, 443, 448, 457, 459, 463, 464, 465, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476,
                    477, 478, 481, 482, 483, 486, 488, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 502, 504, 505,
                    507, 508, 509, 516, 517, 518, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 545, 546,
                    547, 549, 550, 551, 552, 553, 554, 556, 557, 565, 566, 567, 568, 569, 570, 571, 572, 574, 575, 576,
                    579, 580, 581, 582, 583, 585, 586, 587, 590, 591, 592, 593, 594, 596, 597, 599, 601, 602, 603, 604,
                    606, 607, 608, 609, 610, 612, 613, 614, 615, 620, 621, 623, 625, 626, 627, 628, 629, 631, 632, 633,
                    634, 635, 636, 638, 639, 640, 641, 657, 659, 660, 661, 663, 664, 665, 667, 669, 670],
        "taken": [
                    [0, 115, None], [0, 130, None], [0, 145, None], [1, 14, None], [1, 21, None], [1, 28, None],
                    [1, 42, None], [1, 48, None], [1, 52, None], [1, 77, None], [1, 78, None], [1, 81, None],
                    [1, 89, None], [1, 92, None], [1, 97, None], [1, 103, None], [1, 105, None], [1, 113, None],
                    [1, 124, None], [1, 125, None], [1, 126, None], [1, 135, None], [2, 18, 4], [2, 69, 1], [2, 79, 4],
                    [2, 146, 4], [3, 1, 2], [3, 3, 1], [3, 4, 5], [3, 71, 2], [3, 72, 5], [3, 80, 2], [3, 118, 3],
                    [3, 140, 1], [3, 143, 1], [4, 0, 4], [4, 2, None], [4, 44, 1], [4, 47, 3], [4, 50, 3], [4, 54, 3],
                    [4, 58, None], [4, 67, 3], [4, 68, None], [4, 70, 0], [4, 87, 1], [4, 91, 0], [4, 98, 3],
                    [4, 110, 3], [4, 112, 3], [4, 114, 1]],
        "complete_support": 409, "complete_refute": 401, "support": (97, 100), "refute": (99, 100),
        "control_bits": 192_591}
LIBRARY_SOURCES = ("c49_g2.json.gz", "c49_f2.json.gz", "c49_g3.json.gz", *SPEC["earlier"])

g4_report = w6_report


def _key(r: dict) -> tuple:
    return tuple(r["board"]), r["to_move"]


def _label(e: int, r: dict) -> list:
    return [e, r["index"], r.get("root_move")]


def _order(label: list) -> tuple:
    return label[0], label[1], -1 if label[2] is None else label[2]


def g4_selected(earlier: list, g3: dict, assembly: dict) -> tuple:
    """`earlier` is [(run, its SPEC), ...]; returns (the indices of the G3 strategies the assembled game takes, the
    builds it takes as [run's place in `earlier`, index, root move])."""
    labels = [_label(e, r) for e, (run, _) in enumerate(earlier) for r in run["roots"]]
    chosen, _ = v2_assemble([r for run, _ in earlier for r in run["roots"]], assembly)
    index = {_key(r): r["index"] for r in g3["roots"]}
    kept = [k for k in assembly["below"] if k not in chosen]
    kids = {c for k in kept for c in assembly["below"][k] if c in assembly["base"]}
    return sorted(index[c] for c in kids), sorted((labels[n] for n in chosen.values()), key=_order)


def g4_library(library: list, g3: dict, earlier: list, spec: dict) -> list:
    """`library` is G3's; `earlier` the runs in the order of spec["earlier"]."""
    own = {r["index"]: r["found_maps"] for r in g3["roots"]}
    built = {tuple(_label(e, r)): r["found_maps"] for e, run in enumerate(earlier) for r in run["roots"]}
    return ([*library] + [m for i in spec["indices"] for m in own[i]]
            + [m for label in spec["taken"] for m in built[tuple(label)]])


def g4_library_from(g2: dict, f2: dict, g3: dict, *earlier: dict) -> list:
    return g4_library(g3_library(g2, f2), g3, list(earlier), SPEC)


def g4_size(e: dict, earlier: list, g3: dict, assembly: dict, spec: dict) -> dict:
    """`earlier` is [(run, its SPEC), ...]; `assembly` holds J1's opening, below, base, map_cost and G3's library."""
    if _integrity(e, spec) or any(_integrity(run, s) for run, s in earlier):
        return {"verdict": "not_run"}
    rows = [r for run, _ in earlier for r in run["roots"]]
    rank = {k: i for i, k in enumerate(sorted(assembly["below"]))}
    if any(rank.get(_key(r)) != r["index"] for r in rows):
        return {"verdict": "not_run"}
    if not all(j1_valid(r, assembly["library"]) for r in rows):
        return {"verdict": "not_run"}
    before, control = v2_assemble(rows, assembly)
    if control != spec["control_bits"] or g4_selected(earlier, g3, assembly) != (spec["indices"], spec["taken"]):
        return {"verdict": "not_run"}
    at = {r["index"]: _key(r) for r in g3["roots"]}
    if sorted(r["index"] for r in e["roots"]) != spec["indices"] or any(_key(r) != at[r["index"]] for r in e["roots"]):
        return {"verdict": "not_run"}
    library = g4_library(assembly["library"], g3, [run for run, _ in earlier], spec)
    done = [r for r in e["roots"] if r["complete"]]
    if not all(j1_valid(r, library) for r in done):
        return {"verdict": "not_run"}
    level_bits = spec["builder"]["level_bits"]
    cost = {**assembly["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, level_bits)
                                       for i, (m, empty) in enumerate(library, 1) if i > len(assembly["library"])}}
    base = {**assembly["base"], **{_key(r): j1_piece(r, library) for r in done}}
    built = {k: j1_piece(rows[n], assembly["library"]) for k, n in before.items()}

    def size(b):
        return j1_bits(assembly["opening"], assembly["below"], b, built, cost)["bits"]

    held = {_key(r) for r in done}
    while True:
        best = size(base)
        trial = min(((size({**base, k: assembly["base"][k]}), k) for k in sorted(held)), default=(best, None))
        if trial[0] >= best:
            break
        base[trial[1]] = assembly["base"][trial[1]]
        held.discard(trial[1])
    chosen, bits = v2_assemble(rows, {**assembly, "base": base, "map_cost": cost})
    kids = {c for k in assembly["below"] if k not in chosen for c in assembly["below"][k]}
    taken = len(held & kids)
    if bits > control:
        bits, taken = control, 0
    unfinished = len(done) < len(e["roots"])
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    verdict = ("supported" if den * bits <= num * control else
               "refuted" if not unfinished and rden * bits >= rnum * control else "inconclusive")
    return {"verdict": verdict, "bits": bits, "control_bits": control, "ratio": bits / control, "rebuilt": taken,
            "unfinished": unfinished}
