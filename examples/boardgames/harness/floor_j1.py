"""§3.6 J1 — the whole game assembled. E1, O1 and E3 built strategies from ply-6 positions and judged each run
against G3's bits below the same positions, each charging only the library maps G3 never played. J1 puts them into
one game: the opening to ply 6 (harness.opening: 56 moves, 110 trivial positions, 148 frontier positions) and at each
frontier position either G3's piece — the ply-8 opening's move there (4 bits), the trivial positions below it (1 bit
each) and G3's strategies for the rest — or one complete, checked strategy built from the position itself; every
ply-8 position is counted once however many ply-6 positions reach it, and every library map some piece plays is paid
once, so maps nothing plays any more drop out. With no builds this is G3's whole game exactly (280,640 bits). The
choice is made on the whole game's bits: single swaps while any shrinks it, then pairs (two ply-6 positions sharing
the ply-8 positions below them may pay only together), until neither does."""
from __future__ import annotations

from itertools import combinations

from harness.floor_g1 import _coded


def j1_piece(r: dict, library: list) -> tuple:
    return _coded(r), frozenset(n["leaf"] for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library))


def j1_valid(r: dict, library: list) -> bool:
    return bool(r["complete"] and r["checked"] and r["moves_win"] and r["leaves_certified"]
                and r["shared"] == len(library) and r["maps"][1:1 + len(library)] == [m for m, _ in library]
                and r["map_empty"][1:1 + len(library)] == [empty for _, empty in library])


def j1_bits(opening: dict, below: dict, base: dict, built: dict, map_cost: dict) -> dict:
    kept = [k for k in below if k not in built]
    kids = {c for k in kept for c in below[k]}
    roots = {c for c in kids if c in base}
    played = set().union(*(base[c][1] for c in roots), *(p[1] for p in built.values()))
    bits = (opening["moves"] * 4 + opening["trivial"] + 4 * len(kept) + len(kids - roots)
            + sum(base[c][0] for c in roots) + sum(p[0] for p in built.values()) + sum(map_cost[i] for i in played))
    return {"bits": bits, "kept": len(kept), "trivial": len(kids - roots), "strategies": len(roots),
            "library_maps": len(played)}


def j1_choose(opening: dict, below: dict, base: dict, candidates: dict, map_cost: dict) -> tuple:
    """`candidates` maps a frontier position to [(label, piece), ...]; returns ({position: label}, bits)."""
    chosen: dict = {}

    def size(choice):
        return j1_bits(opening, below, base, {k: piece for k, (_, piece) in choice.items()}, map_cost)["bits"]

    def options(k):
        return [None, *candidates[k]]

    def with_(choice, k, option):
        out = {**choice}
        if option is None:
            out.pop(k, None)
        else:
            out[k] = option
        return out

    best = size(chosen)
    keys = sorted(candidates)
    while True:
        moves = [[(k, o)] for k in keys for o in options(k)]
        trial = min(((size(c), c) for c in (with_(chosen, k, o) for [(k, o)] in moves)), key=lambda x: x[0])
        if trial[0] >= best:
            pairs = (with_(with_(chosen, k, o), j, p) for k, j in combinations(keys, 2)
                     for o in options(k) for p in options(j))
            trial = min(((size(c), c) for c in pairs), key=lambda x: x[0], default=(best, chosen))
        if trial[0] >= best:
            return {k: label for k, (label, _) in chosen.items()}, best
        best, chosen = trial
