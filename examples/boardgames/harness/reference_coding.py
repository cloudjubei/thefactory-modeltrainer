"""Exact codes for which map each leaf of a strategy plays (§3.6), read in the order a decoder replays the strategy
(walk_leaves): from the root, a move node's move, then every reply in the game's order, depth first, each position
once. The decoder knows the map list's length and how much of it is the library shared with earlier builds (indices
1..shared; 0 is the empty map), the rest being the build's own maps, each stored once. Codes:

  fixed      — the builder's (harness.strategy_builder): every leaf the index's full width over the whole list;
  first_use  — the library maps the leaves use (the empty map among them) named once — how many, Elias-gamma coded,
               and which of the library — then every leaf an Elias-gamma move-to-front rank in a list that starts as
               a NEW symbol followed by those named maps: NEW reads the build's next own map, so the own maps are
               stored in the order the walk first uses them and each first use costs NEW's rank alone.

first_use needs every own map played by some leaf (a build keeps only those, harness.strategy_builder._compact)."""
from __future__ import annotations

import math

CODES = ("fixed", "first_use")
NEW = -1


def gamma_bits(n: int) -> int:
    """Bits of the Elias-gamma code of a positive integer."""
    if n < 1:
        raise ValueError(f"gamma codes positive integers, not {n}")
    return 2 * (n.bit_length() - 1) + 1


def walk_leaves(game, root, nodes: dict) -> list:
    """The leaves' map indices in the decoder's order; `nodes` maps state keys to {"move"} or {"leaf"} as the builder
    keeps them. A position the strategy reaches without a node raises KeyError."""
    player = game.current_player
    seen: set = set()
    out: list = []
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        node = nodes[key]
        if "leaf" in node:
            out.append(node["leaf"])
            continue
        child = game.step(s, node["move"])
        if game.is_terminal(child):
            continue
        if player(child) == player(s):
            stack.append(child)
            continue
        replies = [c for a in game.legal_actions(child) for c in [game.step(child, a)] if not game.is_terminal(c)]
        stack.extend(reversed(replies))
    return out


def _first_use(refs: list, n_maps: int, shared: int) -> int:
    if any(not 0 <= x < n_maps for x in refs) or not set(range(shared + 1, n_maps)) <= set(refs):
        raise ValueError(f"first_use needs every own map of {shared + 1}..{n_maps - 1} played and no other index")
    named = sorted({x for x in refs if x <= shared})
    bits = gamma_bits(len(named) + 1) + math.ceil(math.log2(math.comb(shared + 1, len(named))))
    order = [NEW, *named]
    met: set = set()
    for x in refs:
        at = order.index(NEW if x > shared and x not in met else x)
        bits += gamma_bits(at + 1)
        order.insert(0, order.pop(at))
        if x > shared and x not in met:
            met.add(x)
            order.insert(1, x)
    return bits


def code_bits(code: str, refs: list, n_maps: int, shared: int) -> int:
    """Bits for the leaf references `refs` (walk order) into a list of `n_maps` maps whose first `shared` after the
    empty map are the library, in `code`."""
    if code == "fixed":
        return len(refs) * (math.ceil(math.log2(n_maps)) if n_maps > 1 else 0)
    if code == "first_use":
        return _first_use(refs, n_maps, shared)
    raise ValueError(f"unknown code {code!r}; one of {CODES}")
