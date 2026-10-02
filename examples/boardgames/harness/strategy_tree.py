"""§C.49 task 6 — the solver-free pieces of the strategy-tree coverage lever.

To play perfectly from the start, a net must be right exactly where its OWN strategy leads against every reply (the
lesson of §C.48: its failures sit on positions it never trained on). `strategy_tree_positions` walks that tree: the
net's single raw move at each of `player`'s positions, every legal reply at the other side's. The process relabels
those positions with its own search and trains on them — no solver anywhere.

`disagreements` is the solver-free reading of how far the net is from its own search there: a raw move DISAGREES when
the search gave it under half the share of its top move — so a different move the search rated about equally is not
counted, the flaw that made §C.48's self-agreement unusable."""
from __future__ import annotations

import random
from typing import Callable

AGREE_SHARE = 0.5


def strategy_tree_positions(game, root, player: int, choose: Callable[[list], list], depth) -> list:
    """Every non-terminal position of `player` within `depth` plies of `root` (None: the whole game) that the raw
    strategy `choose` (a list of states → a list of moves) reaches against every reply — each once, in breadth-first
    order. The chooser is asked once per ply for that ply's positions."""
    if depth is not None and depth < 1:
        raise ValueError(f"strategy tree depth must be >= 1 or None, got {depth}")
    rng = random.Random(0)
    level = {game.state_key(root): root}
    seen: set = set()
    out = []
    ply = 0
    while level and (depth is None or ply < depth):
        fresh = {k: s for k, s in level.items() if k not in seen and not game.is_terminal(s)}
        seen.update(fresh)
        mine = [s for s in fresh.values() if game.current_player(s) == player]
        out.extend(mine)
        nxt: dict = {}
        for s, a in zip(mine, choose(mine) if mine else [], strict=True):
            child = game.step(s, a, rng)
            nxt.setdefault(game.state_key(child), child)
        for s in fresh.values():
            if game.current_player(s) != player:
                for b in game.legal_actions(s):
                    child = game.step(s, b, rng)
                    nxt.setdefault(game.state_key(child), child)
        level = nxt
        ply += 1
    return out


def disagreements(moves: list, labels: list, values: list | None = None, delta: float = 0.0) -> int:
    """How many raw moves got under AGREE_SHARE of the top move's share in their search label. With `values` (one
    {move: search Q, None when unvisited} per position) a move also agrees when its Q is within `delta` of the Q of
    the label's top move — two moves the search values alike are both kept, whatever their shares."""
    if len(moves) != len(labels):
        raise ValueError(f"moves and labels must be the same length, got {len(moves)} and {len(labels)}")
    if values is None:
        return sum(1 for a, pi in zip(moves, labels) if pi[a] < AGREE_SHARE * max(pi))
    if len(values) != len(moves):
        raise ValueError(f"value agreement needs one value row per position, got {len(values)} for {len(moves)}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")

    def agrees(a, pi, q) -> bool:
        if pi[a] >= AGREE_SHARE * max(pi):
            return True
        top = q[pi.index(max(pi))]
        return q[a] is not None and top is not None and q[a] >= top - delta
    return sum(1 for a, pi, q in zip(moves, labels, values) if not agrees(a, pi, q))


def raw_chooser(game, net) -> Callable[[list], list]:
    """The net's RAW move for a batch of positions: the argmax over legal moves of one eval-mode forward pass, no
    gradient; the net is put back in the mode it was in."""
    def choose(states: list) -> list:
        import torch

        from harness.neural import encode

        was = net.training
        net.eval()
        out = []
        for i in range(0, len(states), 4096):
            chunk = states[i:i + 4096]
            legal = torch.zeros(len(chunk), game.num_actions, dtype=torch.bool)
            for j, s in enumerate(chunk):
                legal[j, game.legal_actions(s)] = True
            with torch.no_grad():
                logits, _v = net(torch.stack([encode(game, s) for s in chunk]))
            out += logits.masked_fill(~legal, float("-inf")).argmax(dim=1).tolist()
        net.train(was)
        return out
    return choose
