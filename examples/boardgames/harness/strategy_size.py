"""How small a certified strategy for one side can be (plan §2.3 E2): against the game graph — every reachable
position — a strategy only needs a move at each of its own positions that its own moves and every reply reach, and
only a value-keeping one. Two strategies are built from the game's exact value (`game.position_value`, to the side to
move): the canonical one (the first value-keeping move in action order) and the one whose value-keeping choices
minimise the decisions in its tree (a dynamic programme over positions; a repeated position counts each time it is
reached, so the tree bounds the decisions the strategy really needs — `decisions` counts each position once).
Generic over solvable games; the value of a move is read through the mover of the position it leads to, so a game
whose mover moves again (Kalah) is read correctly."""
from __future__ import annotations

import sys


def game_graph_size(game, root) -> int:
    key = game.state_key
    seen = {key(root)}
    stack = [root]
    while stack:
        s = stack.pop()
        for a in game.legal_actions(s):
            child = game.step(s, a)
            k = key(child)
            if k not in seen:
                seen.add(k)
                stack.append(child)
    return len(seen)


def _kept(game, state, action) -> int:
    child = game.step(state, action)
    me = game.current_player(state)
    if game.is_terminal(child):
        r = game.returns(child)[me]
        return (r > 0) - (r < 0)
    v = game.position_value(child)
    return v if game.current_player(child) == me else -v


def _keeping(game, state) -> list[int]:
    value = game.position_value(state)
    return [a for a in game.legal_actions(state) if _kept(game, state, a) == value]


def canonical_strategy(game, root, player: int) -> dict:
    """{position key: the first value-keeping move in action order} at every position of `player` it reaches."""
    choice: dict = {}
    stack = [root]
    while stack:
        s = stack.pop()
        if game.is_terminal(s):
            continue
        key = game.state_key(s)
        if game.current_player(s) == player:
            if key in choice:
                continue
            choice[key] = _keeping(game, s)[0]
            stack.append(game.step(s, choice[key]))
        else:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
    return choice


def min_tree_strategy(game, root, player: int) -> dict:
    """{"choice": {position key: move}, "tree": decisions in its tree, a repeated position counted each time}, the
    value-keeping choices minimising that tree."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))
    cost: dict = {}
    best: dict = {}

    def tree(s) -> int:
        if game.is_terminal(s):
            return 0
        key = game.state_key(s)
        if key not in cost:
            if game.current_player(s) == player:
                options = [(1 + tree(game.step(s, a)), a) for a in _keeping(game, s)]
                cost[key], best[key] = min(options)
            else:
                cost[key] = sum(tree(game.step(s, b)) for b in game.legal_actions(s))
        return cost[key]

    total = tree(root)
    choice: dict = {}
    stack = [root]
    while stack:
        s = stack.pop()
        if game.is_terminal(s):
            continue
        key = game.state_key(s)
        if game.current_player(s) == player:
            if key in choice:
                continue
            choice[key] = best[key]
            stack.append(game.step(s, choice[key]))
        else:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
    return {"choice": choice, "tree": total}


def decisions(game, root, player: int, choice: dict) -> int:
    """How many distinct positions of `player` the strategy `choice` reaches against every reply."""
    seen: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        if game.is_terminal(s):
            continue
        key = game.state_key(s)
        if game.current_player(s) == player:
            if key in seen:
                continue
            seen.add(key)
            stack.append(game.step(s, choice[key]))
        else:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
    return len(seen)
