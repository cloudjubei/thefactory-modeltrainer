"""§3.6 — what a map must do. Under the rule with ties "lowest", unlevelled cells "last" and order "lowest", a map's
levels matter only through the order they put the safe moves' cells in: at a position where the rule has a choice, the
move it makes must sit on a lower level than every lower-numbered safe move (which would otherwise win the tie) and on
no higher a level than every higher-numbered one; the cells the map leaves without a level read as one more level after
its last. `demands` walks every line a leaf plays and collects those two orders, `strict` and `weak`, as pairs (the
move's cell, the other move's cell), on the board the leaf reads (the mirrored one for a mirrored reading). Any map
keeping every pair makes every move the leaf made, so it wins every line the leaf won.

`lowest` is the map with the fewest levelled cells that keeps a set of pairs, each cell on the lowest level the pairs
allow; `merged` gives the maps whose pairs agree one map, each joining the map so far whose cells it grows the least,
when that costs fewer cells than its own; `shrink` rewrites a strategy with the maps it plans, numbered by content, and
drops each exception the new map now plays by itself."""
from __future__ import annotations

from harness.mirror_builder import board_image
from harness.steady_state import Facts, choose

RULE = {"ties": "lowest", "unlevelled": "last", "order": "lowest"}


def demands(facts: Facts, root, levels: dict, n_levels: int, exceptions: dict | None = None, cells: list | None = None,
            mirrored: bool = False) -> dict:
    """{"strict", "weak", "exceptions": {position key: state} for each exception reached}, from one walk of every line
    the leaf at `root` plays."""
    if any(getattr(facts, k) != v for k, v in RULE.items()):
        raise ValueError(f"the orders are read only under the rule {RULE}")
    exceptions = exceptions or {}
    if mirrored and exceptions:
        raise ValueError("a mirrored reading has no exceptions")
    if mirrored and cells is None:
        raise ValueError("a mirrored reading needs the board's mirror")
    image = board_image(cells) if mirrored else None
    game = facts.game
    player = game.current_player(root)
    strict: set = set()
    weak: set = set()
    reached: dict = {}
    seen: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) != player:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
            continue
        if key in exceptions:
            reached[key] = s
            stack.append(game.step(s, exceptions[key]))
            continue
        read = image(s) if mirrored else s
        move = choose(facts, read, levels, n_levels)
        if move is None:
            raise ValueError(f"the rule has no move at {key}")
        wins, safe, cell = facts.of(read)
        if not wins:
            for other in safe:
                if other != move:
                    (strict if other < move else weak).add((cell[move], cell[other]))
        if mirrored:
            target = cells[facts.placed_cell(read, move)]
            move = next(a for a in game.legal_actions(s) if facts.placed_cell(s, a) == target)
        stack.append(game.step(s, move))
    return {"strict": strict, "weak": weak, "exceptions": reached}


def lowest(strict: set, weak: set, n_levels: int) -> dict | None:
    """{cell: level}, or None when no map of `n_levels` levels keeps the pairs."""
    need = {m for m, _ in strict}
    pairs = strict | weak
    grew = True
    while grew:
        grew = False
        for m, o in pairs:
            if o in need and m not in need:
                need.add(m)
                grew = True
    level = dict.fromkeys(need, 0)
    edges = [(m, o, 1) for m, o in strict if o in need] + [(m, o, 0) for m, o in weak if o in need]
    grew = True
    while grew:
        grew = False
        for m, o, step in edges:
            if level[m] + step > level[o]:
                level[o] = level[m] + step
                if level[o] >= n_levels:
                    return None
                grew = True
    return level


def merged(orders: dict, n_levels: int) -> dict:
    """{map id: its levels}, from {map id: (strict, weak)}. Map 0, the empty map, stays empty and alone."""
    plan: dict = {}
    own: dict = {}
    for i, (strict, weak) in orders.items():
        levels = lowest(strict, weak, n_levels)
        if levels is None:
            raise ValueError(f"map {i}: no map of {n_levels} levels keeps its orders")
        own[i] = levels
    if own.get(0):
        raise ValueError("the empty map's leaves demand levels it does not have")
    if 0 in own:
        plan[0] = {}
    groups: list = []
    for i in sorted((i for i in own if i != 0), key=lambda i: (-len(own[i]), i)):
        strict, weak = orders[i]
        best = None
        for g in groups:
            levels = lowest(g["strict"] | strict, g["weak"] | weak, n_levels)
            if levels is not None and (best is None or len(levels) - len(g["levels"]) < best[1]):
                best = (g, len(levels) - len(g["levels"]), levels)
        if best is not None and best[1] < max(len(own[i]), 1):
            g = best[0]
            g["strict"], g["weak"], g["levels"] = g["strict"] | strict, g["weak"] | weak, best[2]
            g["ids"].append(i)
        else:
            groups.append({"strict": set(strict), "weak": set(weak), "levels": own[i], "ids": [i]})
    for g in groups:
        for i in g["ids"]:
            plan[i] = g["levels"]
    return plan


def shrink(strategy: dict, plan: dict, reached: dict, facts: Facts, n_levels: int, state_of) -> dict:
    """`strategy` with each leaf's map replaced by `plan`'s; `reached` holds the exceptions' states, `state_of(key)`
    makes any other."""
    maps: list = [{}]
    ids: dict = {(False, ()): 0}
    nodes: dict = {}
    for key, node in strategy["nodes"].items():
        if "move" in node:
            nodes[key] = node
            continue
        levels = plan[node["leaf"]]
        mirrored = bool(node.get("mirrored"))
        content = (mirrored and not levels, tuple(sorted(levels.items())))
        if content not in ids:
            ids[content] = len(maps)
            maps.append(dict(levels))
        out = {"leaf": ids[content], **({"mirrored": True} if mirrored else {})}
        kept = {k: move for k, move in node.get("exceptions", {}).items()
                if choose(facts, reached[k] if k in reached else state_of(k), levels, n_levels) != move}
        if kept:
            out["exceptions"] = kept
        nodes[key] = out
    return {"nodes": nodes, "maps": maps}
