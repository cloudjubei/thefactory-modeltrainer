"""The hybrid's EXCEPTION TABLE: knowledge the process computes for itself where self-play cannot learn it.

Self-play keeps improving general play but leaves the Connect-4 opening where it is (h127, h128). The hybrid lets an
exact "solve this position" step cover the opening — only where the net is wrong. `exception_table` walks the
strategy's own tree for one player (its move at that player's positions, every reply at the other side's), and at the
player's positions before `horizon` plies asks for every move's exact value: where the strategy's move is not optimal
it stores an optimal move (the first in legal order) and follows it, so the tree walked is the one the hybrid will
play. `hybrid_chooser` plays the table where it has an entry and the strategy everywhere else. Generic: it needs only
the game's rules, a chooser and an exact move-value oracle; a game can plug in a fast solver."""
from __future__ import annotations

import random
from typing import Callable


def exception_table(game, root, player: int, choose: Callable[[list], list], move_values: Callable[[list], list],
                    horizon: int) -> tuple:
    """({position key: optimal move} where the strategy's move is not optimal, stats) over `player`'s positions in
    the hybrid's own tree within `horizon` plies of `root`. `move_values` maps a list of positions to one
    {move: exact value to the mover} each."""
    if horizon < 1:
        raise ValueError(f"the table horizon must be >= 1 ply, got {horizon}")
    rng = random.Random(0)
    level = {game.state_key(root): root}
    seen: set = set()
    table: dict = {}
    stats = {"positions": 0, "overridden": 0, "positions_by_ply": {}}
    ply = 0
    while level and ply < horizon:
        fresh = {k: s for k, s in level.items() if k not in seen and not game.is_terminal(s)}
        seen.update(fresh)
        mine = [s for s in fresh.values() if game.current_player(s) == player]
        nxt: dict = {}
        if mine:
            stats["positions"] += len(mine)
            stats["positions_by_ply"][str(ply)] = len(mine)
            for s, move, values in zip(mine, choose(mine), move_values(mine), strict=True):
                best = max(values.values())
                if values[move] != best:
                    move = next(a for a in game.legal_actions(s) if values[a] == best)
                    table[game.state_key(s)] = move
                    stats["overridden"] += 1
                child = game.step(s, move, rng)
                nxt.setdefault(game.state_key(child), child)
        for s in fresh.values():
            if game.current_player(s) != player:
                for b in game.legal_actions(s):
                    child = game.step(s, b, rng)
                    nxt.setdefault(game.state_key(child), child)
        level = nxt
        ply += 1
    return table, stats


def hybrid_chooser(game, table: dict, choose: Callable[[list], list]) -> Callable[[list], list]:
    """The table's move where it has one, the strategy's move everywhere else."""
    def hybrid(states: list) -> list:
        return [table.get(game.state_key(s), m) for s, m in zip(states, choose(states), strict=True)]
    return hybrid
