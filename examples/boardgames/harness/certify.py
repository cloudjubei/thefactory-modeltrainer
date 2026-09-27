"""§C.49 — CERTIFY that one player plays perfectly from a root: at every position that player can reach, against
every reply the other side can make, its move keeps the position's exact value. For Connect-4 from the empty board
(a first-player win) this is P-START: a strategy that keeps a proven win at every one of its moves must win, because
a finished draw is not a win.

From a WON root the player's own positions are never solved: the walk follows only moves that keep the win, so
every position the other side faces is lost for it, and every reply leads to a position won for the player — by the
rules, not by assumption. Only the position after each of the player's moves is solved.

The walk visits each distinct position once (transpositions merge), a ply at a time, so the chooser — a net's raw
argmax, say — is asked once per ply for the whole ply. A move that does not keep the value is a FAILURE; the walk
records it and does not follow it (the certificate has already failed there). The walk can be capped by a node
budget; a capped walk is INCOMPLETE and never certifies."""
from __future__ import annotations

import random
from typing import Callable

MAX_EXAMPLES = 20


def certify(game, root, player: int, choose: Callable[[list], list], value_fn: Callable, max_nodes: int | None = None,
            max_depth: int | None = None, value_many: Callable[[list], list] | None = None,
            root_value: int | None = None) -> dict:
    """Walk every position reachable from `root` with `player` moving by `choose` (a list of states → a list of
    moves) and the other side moving every legal way; check each of `player`'s moves keeps `value_fn` (the exact
    value to the side to move). `max_depth` certifies only the first that many plies from the root — the staged
    reading "certified through depth D" when the whole game is out of reach. `value_many` (a list of states → their
    values), when given, receives each ply's solves in one batch — so a caller can spread them over processes.
    `root_value`, when given, is used instead of solving the root — for a root whose value is an established fact
    (Connect-4's empty board: a first-player win) and whose solve is the most expensive of all. Returns {"certified" (every position
    within the horizon walked and none failed), "complete", "complete_game" (nothing was left unwalked at all),
    "horizon", "root_value", "failures", "failures_by_ply" ({depth: count}), "failure_examples", "nodes": {"player", "opponent"}, "by_ply": {depth:
    [player positions, opponent positions]}}."""
    rng = random.Random(0)
    level = {game.state_key(root): root}
    seen: set = set()
    nodes = {"player": 0, "opponent": 0}
    by_ply: dict = {}
    failures, examples, failures_by_ply = 0, [], {}
    complete = True
    if root_value is None and not game.is_terminal(root):
        root_value = int(value_fn(root))
    ply = 0
    while level:
        if max_depth is not None and ply >= max_depth:
            break
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
        pairs = [(s, a, game.step(s, a, rng) if a in game.legal_actions(s) else None)
                 for s, a in zip(mine, choose(mine) if mine else [], strict=True)]
        needed = {} if root_value == 1 else {game.state_key(s): s for s, _a, _c in pairs}
        needed.update({game.state_key(c): c for _s, _a, c in pairs if c is not None and not game.is_terminal(c)})
        keys = list(needed)
        solved = value_many([needed[k] for k in keys]) if value_many and keys else [value_fn(needed[k]) for k in keys]
        values = {k: int(v) for k, v in zip(keys, solved, strict=True)}
        for s, a, child in pairs:
            value = 1 if root_value == 1 else values[game.state_key(s)]
            if child is None:
                kept = None
            elif game.is_terminal(child):
                kept = int(round(game.returns(child)[player]))
            else:
                kept = -values[game.state_key(child)]
            if kept != value:
                failures += 1
                failures_by_ply[ply] = failures_by_ply.get(ply, 0) + 1
                if len(examples) < MAX_EXAMPLES:
                    examples.append({"key": game.state_key(s), "action": a, "value": value,
                                     "kept": -2 if kept is None else kept})
                continue
            nxt.setdefault(game.state_key(child), child)
        for s in theirs:
            for b in game.legal_actions(s):
                child = game.step(s, b, rng)
                nxt.setdefault(game.state_key(child), child)
        level = nxt
        ply += 1
    beyond = any(not game.is_terminal(s) for s in level.values())
    return {"certified": complete and failures == 0, "complete": complete,
            "complete_game": complete and not beyond, "horizon": max_depth, "root_value": root_value,
            "failures": failures, "failures_by_ply": failures_by_ply, "failure_examples": examples, "nodes": nodes, "by_ply": by_ply}
