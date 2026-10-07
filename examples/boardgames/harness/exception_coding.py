"""Exceptions coded in walk order (§3.6): a decoder replays a leaf from its root with the game's rules, the map and the
exceptions read so far, visiting positions in one fixed order. At each of the side's positions it already knows
whether an exception is possible: none where the rule is FORCED (an immediate win, or exactly one safe move — a
winning move is always safe); one for certain where the map gives no move (UNDEFINED); otherwise the position is
CONTESTED and only there is an exception flagged. The flags are coded as how many and which of the contested positions
(an enumerative code), and each exception's move as one of the safe moves the map did not play. Against the sparse
list or mask of harness.steady_exceptions.exception_bits, positions the rules settle cost nothing."""
from __future__ import annotations

import math

from harness.steady_exceptions import choose_with
from harness.steady_state import Facts, choose


def enumerative_bits(m: int, k: int) -> int:
    """Bits to say that k of m items are chosen and which: the count, then the index of the subset."""
    if not 0 <= k <= m:
        raise ValueError(f"cannot choose {k} of {m}")
    return math.ceil(math.log2(math.comb(m, k))) + math.ceil(math.log2(m + 1))


def walk_order_cost(facts: Facts, root, levels: dict, n_levels: int, exceptions: dict) -> dict:
    """{"contested", "flagged" (exceptions at contested positions), "implicit" (exceptions where the map is undefined),
    "move_bits", "bits"} for coding `exceptions` of the leaf at `root` in walk order. Refuses exceptions the decoder
    could never read: at a forced position, with a move that is not safe, or a position left undefined without one."""
    game = facts.game
    player = game.current_player(root)
    seen: set = set()
    stack = [root]
    contested = flagged = implicit = move_bits = 0
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) != player:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
            continue
        wins, safe, _cell = facts.of(s)
        move = choose(facts, s, levels, n_levels)
        fix = exceptions.get(key)
        if fix is not None:
            if wins or len(safe) == 1:
                raise ValueError(f"an exception at {key}, where the rule is forced — a decoder never reads one there")
            if fix not in safe:
                raise ValueError(f"the exception at {key} plays {fix}, which is not safe — it cannot be winning")
            alternatives = len(safe) - (move is not None)
            move_bits += math.ceil(math.log2(alternatives)) if alternatives > 1 else 0
        if move is None:
            if fix is None:
                raise ValueError(f"the map is undefined at {key} and no exception covers it")
            implicit += 1
        elif not wins and len(safe) > 1:
            contested += 1
            flagged += fix is not None
        stack.append(game.step(s, choose_with(facts, s, levels, n_levels, exceptions)))
    return {"contested": contested, "flagged": flagged, "implicit": implicit, "move_bits": move_bits,
            "bits": enumerative_bits(contested, flagged) + move_bits}
