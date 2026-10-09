"""A complete certified strategy for one side from a position, as an exact table plus steady states (plan §3.6 S2).
Depth first from the root, each of the side's positions becomes:

  - a LEAF governed by a priority map (harness.steady_state) when the map wins every line from there: the empty map
    first (the position is trivial), then the maps already found in this build (most recent first — siblings share
    them), then a fresh `search` from the most recent map: a map that wins every line is simplified and added; a map
    plus exceptions (harness.steady_exceptions — table moves where its move does not win) is kept only when the leaf
    is at least `accept` times smaller than its own positions as a 3-bit table — and, with `choose_by_size`, only
    when the whole strategy is no larger with it than with the split below the position (built under the same rule):
    a fixed threshold alone splits some positions to 7x the positions they reach and keeps exceptions others should
    split (h232); a split that runs out of time loses to the leaf, so the choice never costs completion;
  - otherwise a MOVE node: one of its winning moves, chosen to maximise how many of the positions it leads to (after
    every reply) are already trivial or covered by a known map, and the build continues below it.

Once a line reaches a leaf its map plays the rest of the game; each leaf's map was verified from the leaf over every
continuation, so the composite wins wherever play goes. `check` re-walks the whole strategy without the oracle. Size:
one bit per node (move or leaf), three bits per table move, a leaf's reference into the map list, every map's bits
(harness.steady_state.map_bits over the cells empty where it was found) and, once any leaf carries exceptions, a flag
bit per leaf plus each such leaf's exceptions coded in walk order (harness.exception_coding: flags only where the
rules leave a choice, h183).

`search(state, start, seconds)` returns None (nothing found) or {"levels", "exceptions" (empty for a map that wins
every line), "own_positions" (the leaf's own positions; needed only with exceptions)}. `seconds` is the search's
budget at that depth — from `budgets`, [[depth, seconds], ...]: the deepest entry at or above the position's depth, so
shallow positions, whose one map would cover the most, can be searched longer, never past the build's deadline — or
None for the search's default. A `library` of maps found in earlier builds ([(levels, cells empty where found), ...])
is tried like the build's own maps; a build pays only its references to library maps, their bits being paid once for
all the builds sharing them, and `found_maps` hands the maps it adds to the next build. `verifier` checks that a map
wins every line (harness.steady_state.verify, or one with its contract — the C verify for Connect-4); every walk the
builder makes stops at its deadline (h220)."""
from __future__ import annotations

import math
import time

from harness.exception_coding import walk_order_cost
from harness.steady_exceptions import choose_with, verify_with
from harness.steady_state import Facts, map_bits, simplify, verify


class Builder:
    def __init__(self, facts: Facts, winning, search, n_levels: int, level_bits: int, cap: int,
                 min_leaf_depth: int, reuse_window: int, seconds: float, accept: float | None = None,
                 budgets: list | None = None, library: list | None = None, verifier=verify,
                 choose_by_size: bool = False):
        self.facts, self.game = facts, facts.game
        self.winning, self.search = winning, search
        self.n_levels, self.level_bits, self.cap = n_levels, level_bits, cap
        self.min_leaf_depth, self.reuse_window, self.accept = min_leaf_depth, reuse_window, accept
        self.verifier, self.choose_by_size = verifier, choose_by_size
        self.budgets = sorted(budgets or [])
        self.deadline = time.monotonic() + seconds
        self.shared = len(library or [])
        self.maps: list = [{}] + [dict(levels) for levels, _empty in library or []]
        self.map_empty: list = [0] + [empty for _levels, empty in library or []]
        self.nodes: dict = {}
        self.searches = {"tried": 0, "found": 0, "excepted": 0}

    def _covering_map(self, state):
        recent = range(len(self.maps) - 1, max(0, len(self.maps) - 1 - self.reuse_window), -1)
        empty = set(self.facts.empty_cells(state))
        for i in [0, *recent]:
            levels = {c: k for c, k in self.maps[i].items() if c in empty}
            if self.verifier(self.facts, state, levels, self.n_levels, self.cap, deadline=self.deadline)["won"]:
                return i
        return None

    def _children(self, state, move) -> list:
        game = self.game
        child = game.step(state, move)
        if game.is_terminal(child):
            return []
        player = game.current_player(state)
        if game.current_player(child) == player:
            return [child]
        return [c for b in game.legal_actions(child) for c in [game.step(child, b)] if not game.is_terminal(c)]

    def _budget(self, depth: int):
        within = [seconds for d, seconds in self.budgets if d <= depth]
        if not within:
            return None
        return min(within[-1], max(0.0, self.deadline - time.monotonic()))

    def _keep(self, levels: dict, empty: int) -> int:
        self.maps.append(levels)
        self.map_empty.append(empty)
        return len(self.maps) - 1

    def _leaf(self, state, depth: int):
        found = self._covering_map(state)
        if found is not None:
            return {"leaf": found}
        if depth < self.min_leaf_depth:
            return None
        self.searches["tried"] += 1
        empty = self.facts.empty_cells(state)
        start = {c: k for c, k in self.maps[-1].items() if c in set(empty)}
        result = self.search(state, start, self._budget(depth))
        if result is None:
            return None
        levels, exc = result["levels"], result["exceptions"]
        if not exc:
            self.searches["found"] += 1
            kept = simplify(self.facts, state, levels, self.n_levels, self.cap, deadline=self.deadline,
                            verifier=self.verifier)
            return {"leaf": self._keep(kept, len(empty))}
        if self.accept is None:
            return None
        own = result["own_positions"]
        try:
            coded = walk_order_cost(self.facts, state, levels, self.n_levels, exc, deadline=self.deadline)["bits"]
        except TimeoutError:
            return None
        bits = (map_bits(levels, len(empty), self.level_bits) if levels else 0) + coded
        if 3 * own < self.accept * bits:
            return None
        self.searches["excepted"] += 1
        return {"levels": levels, "empty": len(empty), "exceptions": exc, "own": own, "coded": coded}

    def _place(self, key, leaf: dict) -> None:
        if "levels" in leaf:
            index = self._keep(leaf["levels"], leaf["empty"]) if leaf["levels"] else 0
            leaf = {"leaf": index, "exceptions": leaf["exceptions"], "own": leaf["own"], "coded": leaf["coded"]}
        self.nodes[key] = leaf

    def _smaller(self, state, depth: int, leaf: dict) -> bool:
        key = self.game.state_key(state)
        nodes, kept = dict(self.nodes), len(self.maps)
        split = None
        if self._split(state, depth):
            split = (self.nodes, self.maps[kept:], self.map_empty[kept:], self.bits()["nodes"])
        self.nodes = nodes
        del self.maps[kept:], self.map_empty[kept:]
        self._place(key, leaf)
        if split is not None and split[3] < self.bits()["nodes"]:
            self.nodes = split[0]
            self.maps[kept:], self.map_empty[kept:] = split[1], split[2]
        return True

    def build(self, state, depth: int = 0) -> bool:
        """Fill `nodes` for the strategy from `state`; False when the deadline passes first."""
        key = self.game.state_key(state)
        if key in self.nodes:
            return True
        if time.monotonic() > self.deadline:
            return False
        leaf = self._leaf(state, depth)
        if leaf is not None and "levels" in leaf and self.choose_by_size:
            return self._smaller(state, depth, leaf)
        if leaf is not None:
            self._place(key, leaf)
            return True
        return self._split(state, depth)

    def _split(self, state, depth: int) -> bool:
        key = self.game.state_key(state)
        moves = sorted(self.winning([state])[0])
        if not moves:
            raise ValueError("the build reached a position its side cannot win — the oracle or the root is wrong")
        scored = []
        for a in moves:
            children = self._children(state, a)
            ready = sum(1 for c in children if self._covering_map(c) is not None)
            scored.append((-(ready - len(children)), len(children), a, children))
        _gap, _n, move, children = min(scored)
        self.nodes[key] = {"move": move}
        return all(self.build(c, depth + 2) for c in children)

    def bits(self) -> dict:
        moves = sum(1 for n in self.nodes.values() if "move" in n)
        leaves = len(self.nodes) - moves
        ref = math.ceil(math.log2(len(self.maps))) if len(self.maps) > 1 else 0
        own = 1 + self.shared
        maps = sum(map_bits(m, e, self.level_bits) for m, e in zip(self.maps[own:], self.map_empty[own:]))
        move_bits = math.ceil(math.log2(self.game.num_actions))
        excepted = [n for n in self.nodes.values() if n.get("exceptions")]
        exc_bits = leaves + sum(n["coded"] for n in excepted) if excepted else 0
        return {"nodes": len(self.nodes) + moves * move_bits + leaves * ref + maps + exc_bits, "moves": moves,
                "leaves": leaves, "maps": len(self.maps) - own, "map_bits": maps,
                "exceptions": sum(len(n["exceptions"]) for n in excepted), "exception_bits": exc_bits}


    def found_maps(self) -> list:
        """The maps this build added, as (levels, cells empty where found) — the next build's library."""
        own = 1 + self.shared
        return list(zip(self.maps[own:], self.map_empty[own:]))


def check(facts: Facts, root, nodes: dict, maps: list, n_levels: int, cap: int) -> dict:
    """Walk the strategy from `root` against every reply without any oracle: every reached position of the side must
    be a node; a move node's move is played on; a leaf's map, with its exceptions, must win every line from there.
    Returns {"won", "reason",
    "own_positions" (distinct positions of the side the composite strategy reaches, leaves' subtrees included)}."""
    game = facts.game
    player = game.current_player(root)
    seen: set = set()
    own: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if game.is_terminal(s):
            if game.winner(s) != player:
                return {"won": False, "reason": "lost line", "own_positions": len(own)}
            continue
        if game.current_player(s) != player:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
            continue
        node = nodes.get(key)
        if node is None:
            return {"won": False, "reason": "missing node", "own_positions": len(own)}
        own.add(key)
        if "move" in node:
            stack.append(game.step(s, node["move"]))
            continue
        empty = set(facts.empty_cells(s))
        levels = {c: k for c, k in maps[node["leaf"]].items() if c in empty}
        exc = node.get("exceptions", {})
        walk = verify_with(facts, s, levels, n_levels, exc, cap)
        if not walk["won"]:
            return {"won": False, "reason": f"leaf {walk['reason']}", "own_positions": len(own)}
        own.update(_own_under(facts, s, levels, n_levels, exc))
    return {"won": True, "reason": None, "own_positions": len(own)}


def _own_under(facts: Facts, root, levels: dict, n_levels: int, exceptions: dict) -> set:
    game = facts.game
    player = game.current_player(root)
    seen, own, stack = set(), set(), [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) == player:
            own.add(key)
            stack.append(game.step(s, choose_with(facts, s, levels, n_levels, exceptions)))
        else:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
    return own
