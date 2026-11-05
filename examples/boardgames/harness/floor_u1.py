"""§3.6 U1 — choosing among G4's rebuilds. G4 (harness.floor_g4) put every complete rebuild in at once and sent back,
one at a time, the rebuild whose return shrank the game most: 76 of 409 stay, 184,556 bits (h358, h359). A rebuild
that plays a new map pays for it alone unless another kept rebuild plays it too, so two rebuilds sharing a map can be
worth keeping together and not apart, and sending them back one at a time cannot drop the map. U1 searches over the
set of rebuilds kept: a move puts one rebuild in or out, or every rebuild playing one map (two or more) in together,
or out together; the move that shrinks the whole game most is taken until none does (`u1_choose`). It starts from
none kept and from all kept, keeps the smaller, then chooses the ply-6 builds again as floor_v2 does where that
shrinks the game (floor_v2 chooses afresh, so it can do worse than the choice the search ran with) — never worse than
G4's game. A finding measured after the data, on G4's evidence; no new builds."""
from __future__ import annotations

from harness.floor_g4 import _key, g4_library, g4_size
from harness.floor_j1 import j1_bits, j1_piece
from harness.floor_v2 import v2_assemble
from harness.steady_state import map_bits


def u1_choose(size, plays: dict, start: frozenset) -> tuple:
    """`size` gives the whole game's bits for a set of kept rebuilds; `plays` maps a rebuild to the maps it plays.
    Returns (the kept set, its bits)."""
    held, best = frozenset(start), size(frozenset(start))
    maps = sorted(set().union(*plays.values()))
    while True:
        moves = {frozenset({k}) for k in plays}
        for m in maps:
            playing = {k for k in plays if m in plays[k]}
            moves |= {group for group in (frozenset(playing - held), frozenset(playing & held)) if len(group) > 1}
        trial = min(((size(held ^ move), sorted(move)) for move in moves), default=(best, None))
        if trial[0] >= best:
            return held, best
        held, best = held ^ frozenset(trial[1]), trial[0]


def u1_size(e: dict, earlier: list, g3: dict, assembly: dict, spec: dict) -> dict:
    """G4's judge first (harness.floor_g4.g4_size with G4's SPEC); `from_none` and `from_all` are each start's bits
    before the ply-6 builds are chosen again, `kept` the rebuilds the game takes."""
    g4 = g4_size(e, earlier, g3, assembly, spec)
    if g4["verdict"] == "not_run":
        return {"verdict": "not_run"}
    rows = [r for run, _ in earlier for r in run["roots"]]
    before, _ = v2_assemble(rows, assembly)
    library = g4_library(assembly["library"], g3, [run for run, _ in earlier], spec)
    level_bits = spec["builder"]["level_bits"]
    cost = {**assembly["map_cost"], **{i: map_bits({int(c): k for c, k in m.items()}, empty, level_bits)
                                       for i, (m, empty) in enumerate(library, 1) if i > len(assembly["library"])}}
    rebuilds = {_key(r): j1_piece(r, library) for r in e["roots"] if r["complete"]}
    built = {k: j1_piece(rows[n], assembly["library"]) for k, n in before.items()}

    def size(held):
        base = {**assembly["base"], **{k: rebuilds[k] for k in held}}
        return j1_bits(assembly["opening"], assembly["below"], base, built, cost)["bits"]

    plays = {k: piece[1] for k, piece in rebuilds.items()}
    starts = [u1_choose(size, plays, frozenset()), u1_choose(size, plays, frozenset(rebuilds))]
    held, _ = min(starts, key=lambda s: s[1])
    base = {**assembly["base"], **{k: rebuilds[k] for k in held}}
    chosen, bits = v2_assemble(rows, {**assembly, "base": base, "map_cost": cost})
    if size(held) < bits:
        chosen, bits = before, size(held)
    kids = {c for k in assembly["below"] if k not in chosen for c in assembly["below"][k]}
    kept = len(held & kids)
    if bits > g4["bits"]:
        bits, kept = g4["bits"], g4["rebuilt"]
    return {"verdict": "measured", "bits": bits, "g4_bits": g4["bits"], "kept": kept, "from_none": starts[0][1],
            "from_all": starts[1][1]}
