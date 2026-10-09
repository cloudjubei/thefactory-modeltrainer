"""An opening for one side down to a fixed depth (§3.6, the whole-game strategy): at each of the side's positions
shallower than `plies`, one winning move — the one whose replies the empty map already wins most often (fewest
replies, then the lowest move, on a tie), the builder's readiness rule with no maps found yet — and every reply
followed. Positions the empty map wins need no move; the side's positions at the depth that it does not win are the
frontier, each needing its own strategy (harness.strategy_builder). The whole strategy is the opening's moves plus
those strategies."""
from __future__ import annotations

from harness.steady_state import Facts, verify


def frontier(facts: Facts, root, winning, plies: int, n_levels: int, cap: int) -> dict:
    """{"moves": {position key: move}, "trivial": [keys of positions the empty map wins], "frontier": [the side's
    positions `plies` below the root that it does not]}. Refuses a root its side cannot win."""
    game = facts.game
    if not winning([root])[0]:
        raise ValueError("the root is a position its side cannot win — no opening can")
    player = game.current_player(root)
    moves: dict = {}
    trivial: dict = {}
    found: dict = {}
    seen: set = set()
    stack = [(root, 0)]
    while stack:
        s, depth = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) != player:
            stack.extend((game.step(s, b), depth + 1) for b in game.legal_actions(s))
            continue
        if verify(facts, s, {}, n_levels, cap)["won"]:
            trivial[key] = True
            continue
        if depth >= plies:
            found[key] = s
            continue
        best = None
        for a in sorted(winning([s])[0]):
            after = game.step(s, a)
            replies = [] if game.is_terminal(after) else [game.step(after, b) for b in game.legal_actions(after)]
            ready = sum(1 for c in replies if game.is_terminal(c) or verify(facts, c, {}, n_levels, cap)["won"])
            score = (-(ready - len(replies)), len(replies), a)
            if best is None or score < best[0]:
                best = (score, a, after)
        _score, move, after = best
        moves[key] = move
        stack.append((after, depth + 1))
    return {"moves": moves, "trivial": list(trivial), "frontier": list(found.values())}
