"""§C.49 — CERTIFY that one player plays perfectly from a root: at every position that player can reach, against
every reply the other side can make, its move keeps the position's exact value. For Connect-4 from the empty board
(a first-player win) this is P-START: a strategy that keeps a proven win at every one of its moves must win, because
a finished draw is not a win.

The walk visits each distinct position once (transpositions merge), a ply at a time, so the chooser — a net's raw
argmax, say — is asked once per ply for the whole ply. A move that does not keep the value is a FAILURE; the walk
records it and does not follow it (the certificate has already failed there). The walk can be capped by a node
budget; a capped walk is INCOMPLETE and never certifies."""
from __future__ import annotations

import random
from typing import Callable

MAX_EXAMPLES = 20


def _value_after(game, state, action, mover: int, value_fn) -> int:
    child = game.step(state, action, random.Random(0))
    if game.is_terminal(child):
        return int(round(game.returns(child)[mover]))
    return -int(value_fn(child))


def certify(game, root, player: int, choose: Callable[[list], list], value_fn: Callable, max_nodes: int | None = None
            ) -> dict:
    """Walk every position reachable from `root` with `player` moving by `choose` (a list of states → a list of
    moves) and the other side moving every legal way; check each of `player`'s moves keeps `value_fn` (the exact
    value to the side to move). Returns {"certified", "complete", "root_value", "failures", "failure_examples",
    "nodes": {"player", "opponent"}, "by_ply": {ply: [player positions, opponent positions]}}."""
    rng = random.Random(0)
    level = {game.state_key(root): root}
    seen: set = set()
    nodes = {"player": 0, "opponent": 0}
    by_ply: dict = {}
    failures, examples = 0, []
    complete = True
    root_value = None if game.is_terminal(root) else int(value_fn(root))
    ply = 0
    while level:
        fresh = {k: s for k, s in level.items() if k not in seen and not game.is_terminal(s)}
        if max_nodes is not None and sum(nodes.values()) + len(fresh) > max_nodes:
            complete = False
            break
        seen.update(fresh)
        mine = [s for s in fresh.values() if game.current_player(s) == player]
        theirs = [s for s in fresh.values() if game.current_player(s) != player]
        nodes["player"] += len(mine)
        nodes["opponent"] += len(theirs)
        by_ply[ply] = [len(mine), len(theirs)]
        nxt: dict = {}
        for s, a in zip(mine, choose(mine) if mine else [], strict=True):
            value = int(value_fn(s))
            kept = _value_after(game, s, a, player, value_fn) if a in game.legal_actions(s) else None
            if kept != value:
                failures += 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append({"key": game.state_key(s), "action": a, "value": value,
                                     "kept": -2 if kept is None else kept})
                continue
            child = game.step(s, a, rng)
            nxt.setdefault(game.state_key(child), child)
        for s in theirs:
            for b in game.legal_actions(s):
                child = game.step(s, b, rng)
                nxt.setdefault(game.state_key(child), child)
        level = nxt
        ply += 1
    return {"certified": complete and failures == 0, "complete": complete, "root_value": root_value,
            "failures": failures, "failure_examples": examples, "nodes": nodes, "by_ply": by_ply}
