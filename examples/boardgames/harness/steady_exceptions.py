"""A priority map plus exceptions (§3.6): any map becomes a complete leaf when each of the side's positions whose map
move is not a winning one (or that has no move) gets a table move instead, the map resuming below it. `needed` finds
those positions in one walk with the search's oracle (`winning`); `verify_with` checks the result over every line
without it. Exceptions cost the cheaper of their positions' indices among the leaf's own positions or a mask over
them, plus the moves — the hybrid's table-plus-exceptions accounting inside a leaf (h170: a score that stops at the
first failure hides the rest of the tree)."""
from __future__ import annotations

import math
import time

from harness.steady_state import Facts, choose

CHECK_EVERY = 1024


def choose_with(facts: Facts, state, levels: dict, n_levels: int, exceptions: dict):
    key = facts.game.state_key(state)
    return exceptions[key] if key in exceptions else choose(facts, state, levels, n_levels)


def _walk(facts: Facts, root, levels: dict, n_levels: int, exceptions: dict, cap: int):
    """("clean" | "failed" | "cap", reason, the side's positions on the first failing line from the root)."""
    game = facts.game
    player = game.current_player(root)
    owner: dict = {}
    seen: set = set()
    stack = [(root, None)]
    while stack:
        s, last_own = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > cap:
            return "cap", "cap", []
        if game.is_terminal(s):
            if game.winner(s) != player:
                return "failed", "draw" if game.winner(s) is None else "loss", _line(owner, last_own)
            continue
        if game.current_player(s) == player:
            owner[key] = (s, last_own)
            move = choose_with(facts, s, levels, n_levels, exceptions)
            if move is None:
                return "failed", "undefined", _line(owner, key)
            stack.append((game.step(s, move), key))
        else:
            stack.extend((game.step(s, b), last_own) for b in game.legal_actions(s))
    return "clean", None, []


def _line(owner: dict, key) -> list:
    line = []
    while key is not None:
        s, key = owner[key]
        line.append(s)
    return line[::-1]


def verify_with(facts: Facts, root, levels: dict, n_levels: int, exceptions: dict, cap: int) -> dict:
    status, reason, _line_ = _walk(facts, root, levels, n_levels, exceptions, cap)
    return {"won": status == "clean", "reason": reason}


def needed(facts: Facts, root, levels: dict, n_levels: int, winning, cap: int, deadline: float | None = None) -> dict:
    """{"status": "ok" | "cap" | "timeout", "exceptions": {position key: move}, "own_positions": the side's positions
    walked} — one walk over every line from `root`: where the map's move is not one of `winning`'s moves (or it gives
    none), an exception plays the smallest winning move and the walk follows it, so the leaf is complete by
    construction. The oracle is asked only where the map's move is not an immediate win. "timeout" when
    time.monotonic() passes `deadline` mid-walk (a walk near the root can take minutes, h200)."""
    if not winning([root])[0]:
        raise ValueError("the root is a position its side cannot win — no map or exception can")
    game = facts.game
    player = game.current_player(root)
    exceptions: dict = {}
    seen: set = set()
    own = 0
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > cap:
            return {"status": "cap", "exceptions": {}, "own_positions": own}
        if deadline is not None and len(seen) % CHECK_EVERY == 1 % CHECK_EVERY and time.monotonic() > deadline:
            return {"status": "timeout", "exceptions": {}, "own_positions": own}
        if game.is_terminal(s):
            if game.winner(s) != player:
                raise RuntimeError("a line the oracle's winning moves lead to is not won — the oracle is wrong")
            continue
        if game.current_player(s) == player:
            own += 1
            move = choose(facts, s, levels, n_levels)
            if move is None or move not in facts.of(s)[0]:
                wins = winning([s])[0]
                if not wins:
                    raise RuntimeError("a position the oracle's winning moves reach has no winning move — the oracle "
                                       "is wrong")
                if move not in wins:
                    move = min(wins)
                    exceptions[key] = move
            stack.append(game.step(s, move))
        else:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
    return {"status": "ok", "exceptions": exceptions, "own_positions": own}


def patch(facts: Facts, root, levels: dict, n_levels: int, winning, cap: int, max_exceptions: int) -> dict:
    """{"status": "patched" | "too_many" | "cap", "exceptions": {position key: move}} — the exceptions `needed` finds,
    refused when there are more than `max_exceptions`."""
    r = needed(facts, root, levels, n_levels, winning, cap)
    if r["status"] == "cap":
        return {"status": "cap", "exceptions": {}}
    if len(r["exceptions"]) > max_exceptions:
        return {"status": "too_many", "exceptions": {}}
    return {"status": "patched", "exceptions": r["exceptions"]}


def exception_bits(exceptions: int, positions: int, actions: int) -> int:
    """The cheaper of a sparse list (each exception's index among the leaf's own positions plus its move) and a mask
    (one bit per own position plus each exception's move)."""
    if exceptions == 0:
        return 0
    move = math.ceil(math.log2(actions))
    return min(exceptions * (math.ceil(math.log2(positions)) + move), positions + exceptions * move)
