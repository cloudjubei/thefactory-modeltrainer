"""§3.7 — a composed strategy: harness.strategy_file's nodes and maps, where a leaf's map plays only until the game
reaches one of the side's positions that has a node of its own, which then decides. A move node plays its move, and
every position of the side after it must have a node; a leaf hands play to its map. Proofs built from the bottom up
compose this way: a map at a shallow position need not win the lines a deeper proof already wins. `check` plays every
line against every reply, with no solver and nothing from the prover that built the strategy."""
from __future__ import annotations

from harness.steady_state import Facts, choose


def check(facts: Facts, root, strategy: dict, n_levels: int, cap: int) -> dict:
    """{"won", "reason": None | "lost" (a line ends in a draw or a loss) | "undefined" (a position of the side with no
    node and no map, or a map that gives no move) | "cap"}."""
    game = facts.game
    player = game.current_player(root)
    nodes, maps = strategy["nodes"], strategy["maps"]
    seen: set = set()
    stack = [(root, None)]
    while stack:
        s, current = stack.pop()
        key = game.state_key(s)
        if (key, current) in seen:
            continue
        seen.add((key, current))
        if len(seen) > cap:
            return {"won": False, "reason": "cap"}
        if game.is_terminal(s):
            if game.winner(s) != player:
                return {"won": False, "reason": "lost"}
            continue
        if game.current_player(s) != player:
            stack.extend((game.step(s, b), current) for b in game.legal_actions(s))
            continue
        node = nodes.get(key)
        if node is not None and ("mirrored" in node or "exceptions" in node):
            raise ValueError(f"a composed strategy has no mirrored readings or exceptions: {node}")
        if node is not None and "move" in node:
            if node["move"] not in game.legal_actions(s):
                raise ValueError(f"the node at {key} plays {node['move']}, which is not a legal move")
            move, current = node["move"], None
        else:
            if node is not None:
                current = node["leaf"]
            move = None if current is None else choose(facts, s, maps[current], n_levels)
            if move is None:
                return {"won": False, "reason": "undefined"}
        stack.append((game.step(s, move), current))
    return {"won": True, "reason": None}
