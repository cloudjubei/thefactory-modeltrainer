"""§3.6 J2 — the whole game assembled by transposition group. J1 (harness.floor_j1) swaps one or two ply-6 positions
at a time, but ply-6 positions reached by different move orders share the ply-8 positions below them: G3's strategy at
a shared ply-8 position stays while any ply-6 position above it keeps G3's piece, so a build there saves at most the
4-bit move unless every position sharing it swaps too (h352). J2 joins the ply-6 positions sharing any ply-8 position
into groups (transitively) and, from a starting choice (J1's), tries for each group every subset of its positions with
builds swapped at once, each swapped position taking its cheapest build counting the library maps no position that
cannot swap there already plays; each trial is judged on the whole game's bits (harness.floor_j1.j1_bits), a group's
best kept only if it shrinks the game, and the groups are gone over again until none does. Groups touch only through
library maps, so this is exact within a group up to the build each swapped position takes."""
from __future__ import annotations

from itertools import combinations

from harness.floor_j1 import j1_bits


def j2_groups(below: dict) -> list:
    parent = {k: k for k in below}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    owner: dict = {}
    for k in sorted(below):
        for c in below[k]:
            if c in owner:
                parent[find(k)] = find(owner[c])
            else:
                owner[c] = k
    groups: dict = {}
    for k in sorted(below):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values())


def j2_choose(opening: dict, below: dict, base: dict, candidates: dict, map_cost: dict, start: dict,
              max_swapped: int = 16) -> tuple:
    """`candidates` maps a frontier position to [(label, piece), ...] and `start` a position to a label; returns
    ({position: label}, bits)."""
    pieces = {k: dict(options) for k, options in candidates.items()}
    groups = [swappable for swappable in ([k for k in g if k in candidates] for g in j2_groups(below)) if swappable]
    for swappable in groups:
        if len(swappable) > max_swapped:
            raise ValueError(f"a group has {len(swappable)} positions with builds, over {max_swapped}")

    def size(choice):
        return j1_bits(opening, below, base, {k: pieces[k][label] for k, label in choice.items()}, map_cost)["bits"]

    chosen = dict(start)
    best = size(chosen)
    improved = True
    while improved:
        improved = False
        for swappable in groups:
            rest = {k: label for k, label in chosen.items() if k not in swappable}
            paid = set().union(*(base[c][1] for k in below if k not in swappable and k not in rest
                                 for c in below[k] if c in base),
                               *(pieces[k][label][1] for k, label in rest.items()))

            def cheapest(k):
                return min(candidates[k], key=lambda o: o[1][0] + sum(map_cost[i] for i in o[1][1] - paid))[0]

            current = {k: chosen[k] for k in swappable if k in chosen}
            trial_best, trial = best, None
            for n in range(len(swappable) + 1):
                for subset in combinations(swappable, n):
                    choice = {**rest, **{k: cheapest(k) for k in subset}}
                    if {k: choice[k] for k in swappable if k in choice} == current:
                        continue
                    bits = size(choice)
                    if bits < trial_best:
                        trial_best, trial = bits, choice
            if trial is not None:
                best, chosen, improved = trial_best, trial, True
    return chosen, best
