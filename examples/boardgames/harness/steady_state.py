"""Steady states (§3.6 S1): a compact rule that plays the rest of a placement game from one position, for the side to
move there, after WeakC4 (2swap; https://2swap.github.io/WeakC4/explanation/). The rule is a priority map — a level
0..levels-1 on some of the cells empty at that position — read the same way at every later position:

  1. play a move that wins at once (the lowest such action);
  2. keep the SAFE moves — those after which the opponent cannot win at once; if exactly one is safe, play it;
  3. otherwise go through the levels in order and play the safe move whose cell is at the first level holding
     exactly one safe move; a level holding none or several is passed over;
  4. otherwise the rule has no move: it is undefined there.

A map may have two phases: its levels on cells (keys 0 and up) are read before a SWITCH — a number of stones on the
board — and its second levels (on keys `phase_cell(cell)`, -2 and down) from the switch on. A map without a switch has
one phase.

A move's cell is read from the game's own observation (the one board cell it fills), so the language knows nothing
about any particular game; it applies to placement games, where every move fills exactly one cell.

`verify` walks every position the rule reaches against every reply and accepts only if every line ends in a win for
the side the rule plays — an undefined position, a draw or a loss rejects it. It never consults a solver. Facts that
do not depend on the map (immediate wins, safe moves, their cells) are cached per position across calls, in a cache
emptied whenever it reaches its limit (~1 KB a position: a 4-hour build reached 9.6 GB a worker unbounded)."""
from __future__ import annotations

import time


TIES = ("exact", "lowest")
UNLEVELLED = ("none", "last")
ORDERS = ("lowest", "tactics")
SWITCH = -1


def phase_cell(cell: int) -> int:
    """The key a map's second phase levels `cell` under."""
    return -cell - 2


def restrict(levels: dict, empty) -> dict:
    """`levels` on the cells in `empty` (both phases), its switch kept."""
    empty = set(empty)
    return {c: k for c, k in levels.items() if c == SWITCH or (c if c >= 0 else phase_cell(c)) in empty}


def _phase(facts, state, levels: dict) -> dict:
    if SWITCH in levels and facts.cells - len(facts.empty_cells(state)) >= levels[SWITCH]:
        return {phase_cell(c): k for c, k in levels.items() if c < SWITCH}
    return levels


class Facts:
    """Per-position facts for one game, independent of any priority map: immediate wins, safe moves, each safe
    move's cell. Shared by the verifier and the search so a position is analysed once (until the cache is emptied).
    `ties` is how the rule reads a level (`choose`), carried here because every walk that applies the rule has these
    facts: "exact" — a level decides only when it holds exactly one safe move; "lowest" — the first level holding any
    safe move decides, with its lowest-numbered move. `safe_depth` is what makes a move safe: 1 — the opponent cannot
    win at once after it; 2 — nor can the opponent answer it with a move after which every reply of ours lets them win
    at once (unless none of the 1-ply safe moves passes, when those are kept). `forcing` keeps, of several safe moves,
    those that FORCE — that leave the opponent at most one reply after which the mover cannot win at once (a move that
    ends the game forces nothing) — when any does, all of them otherwise. `unlevelled` is what the rule makes of the
    safe moves on cells the map leaves without a level: "none" — passes them over; "last" — reads them as one more level
    after the map's, by `ties` like any other. `order` is which move a deciding level holding several plays (so only
    with `ties` "lowest"): "lowest" — the lowest-numbered; "tactics" — the lowest of those that win by force in two,
    else of those that do not lose by force in two (`_loses_in_two`), else the lowest."""

    def __init__(self, game, limit: int = 2_000_000, ties: str = "exact", safe_depth: int = 1, forcing: bool = False,
                 unlevelled: str = "none", order: str = "lowest"):
        if ties not in TIES:
            raise ValueError(f"ties must be one of {TIES}, got {ties!r}")
        if safe_depth not in (1, 2):
            raise ValueError(f"safe_depth must be 1 or 2, got {safe_depth!r}")
        if not isinstance(forcing, bool):
            raise ValueError(f"forcing must be True or False, got {forcing!r}")
        if unlevelled not in UNLEVELLED:
            raise ValueError(f"unlevelled must be one of {UNLEVELLED}, got {unlevelled!r}")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
        if order != "lowest" and ties != "lowest":
            raise ValueError(f"order {order!r} breaks ties, so it needs ties 'lowest', got {ties!r}")
        self.game = game
        self.ties = ties
        self.safe_depth = safe_depth
        self.forcing = forcing
        self.unlevelled = unlevelled
        self.order = order
        h, w = getattr(game, "board_shape")
        self.cells = h * w
        self.limit = limit
        self._facts: dict = {}
        self.cleared = 0

    def placed_cell(self, state, action) -> int:
        game = self.game
        mover = game.current_player(state)
        before = game.observation(state, mover)[: self.cells]
        after = game.observation(game.step(state, action), mover)[: self.cells]
        changed = [i for i in range(self.cells) if before[i] != after[i]]
        if len(changed) != 1 or before[changed[0]] != 0:
            raise ValueError(f"{game.name}: action {action} does not fill exactly one empty cell — not a placement move")
        return changed[0]

    def empty_cells(self, state) -> list[int]:
        mover = self.game.current_player(state)
        return [i for i, v in enumerate(self.game.observation(state, mover)[: self.cells]) if v == 0]

    def _wins_now(self, state, action) -> bool:
        game = self.game
        child = game.step(state, action)
        return game.is_terminal(child) and game.winner(child) == game.current_player(state)

    def _loses_in_two(self, state, action) -> bool:
        """Whether the opponent can answer `action` with a move after which every reply of the mover's either loses at
        once (the opponent then wins immediately) or is not available — no reply wins, draws or holds."""
        game = self.game
        child = game.step(state, action)
        if game.is_terminal(child) or game.current_player(child) == game.current_player(state):
            return False
        for b in game.legal_actions(child):
            after = game.step(child, b)
            if game.is_terminal(after):
                continue
            held = False
            for c in game.legal_actions(after):
                reply = game.step(after, c)
                if game.is_terminal(reply) or not any(self._wins_now(reply, d) for d in game.legal_actions(reply)):
                    held = True
                    break
            if not held:
                return True
        return False

    def _wins_in_two(self, state, action) -> bool:
        """Whether every answer of the opponent's to `action` leaves the mover a move that wins at once (an action or
        answer that ends the game wins nothing in two)."""
        game = self.game
        child = game.step(state, action)
        if game.is_terminal(child) or game.current_player(child) == game.current_player(state):
            return False
        for b in game.legal_actions(child):
            after = game.step(child, b)
            if not any(self._wins_now(after, d) for d in game.legal_actions(after)):
                return False
        return True

    def _forces(self, state, action) -> bool:
        """Whether the opponent has at most one answer to `action` after which the mover cannot win at once."""
        game = self.game
        child = game.step(state, action)
        if game.is_terminal(child) or game.current_player(child) == game.current_player(state):
            return False
        safe_replies = 0
        for b in game.legal_actions(child):
            after = game.step(child, b)
            if not any(self._wins_now(after, d) for d in game.legal_actions(after)):
                safe_replies += 1
        return safe_replies <= 1

    def of(self, state) -> tuple:
        """(winning actions, the safe actions the rule chooses among — the forcing ones where they are preferred and
        any force — {such action: its cell}) at `state`."""
        return self._entry(state)[:3]

    def safe_moves(self, state) -> list:
        """Every safe action at `state` (at `safe_depth`), whether or not forcing ones are preferred — the moves an
        exception may play, a winning move always among them."""
        return self._entry(state)[3]

    def pick(self, state, moves: list):
        """The move a deciding level holding `moves` (safe moves at `state`) plays, by `order`."""
        if self.order == "lowest":
            return min(moves)
        return min(moves, key=self._entry(state)[4].__getitem__)

    def _entry(self, state) -> tuple:
        key = self.game.state_key(state)
        if key not in self._facts:
            game = self.game
            mover = game.current_player(state)
            legal = game.legal_actions(state)
            wins = [a for a in legal if self._wins_now(state, a)]
            safe = []
            for a in legal:
                child = game.step(state, a)
                if game.is_terminal(child):
                    safe.append(a)
                elif game.current_player(child) == mover or not any(
                        self._wins_now(child, b) for b in game.legal_actions(child)):
                    safe.append(a)
            if self.safe_depth == 2 and len(safe) > 1:
                safe = [a for a in safe if not self._loses_in_two(state, a)] or safe
            chosen = safe
            if self.forcing and len(safe) > 1:
                chosen = [a for a in safe if self._forces(state, a)] or safe
            if len(self._facts) >= self.limit:
                self._facts.clear()
                self.cleared += 1
            ranks = ({a: (not self._wins_in_two(state, a), self._loses_in_two(state, a), a) for a in chosen}
                     if self.order == "tactics" and len(chosen) > 1 else {})
            self._facts[key] = (wins, chosen, {a: self.placed_cell(state, a) for a in chosen}, safe, ranks)
        return self._facts[key]


def choose(facts: Facts, state, levels: dict, n_levels: int):
    """The rule's move at `state` under the priority map `levels` ({cell: level}, in the phase `state` is in), or None
    where it is undefined: a winning move; else the only safe move; else the first level that decides (`facts.ties`),
    its move by `facts.order` — the safe moves on unlevelled cells a last level when `facts.unlevelled` is "last"."""
    wins, safe, cell = facts.of(state)
    if wins:
        return min(wins)
    if len(safe) == 1:
        return safe[0]
    levels = _phase(facts, state, levels)
    for k in range(n_levels):
        at = [a for a in safe if levels.get(cell[a]) == k]
        if len(at) == 1 or (at and facts.ties == "lowest"):
            return facts.pick(state, at)
    if facts.unlevelled == "last":
        at = [a for a in safe if cell[a] not in levels]
        if len(at) == 1 or (at and facts.ties == "lowest"):
            return facts.pick(state, at)
    return None


def verify(facts: Facts, root, levels: dict, n_levels: int, cap: int = 1_000_000,
           deadline: float | None = None) -> dict:
    """Whether the rule wins every line from `root` for the side to move there. Returns {"won", "reason" (None, or
    the first failure: "undefined", "draw", "loss", "cap", "timeout" — time.monotonic() past `deadline` mid-walk),
    "own_positions" (the rule's decision positions walked), "positions" (all positions walked)}."""
    game = facts.game
    player = game.current_player(root)
    seen: set = set()
    stack = [root]
    own = 0

    def result(reason):
        return {"won": reason is None, "reason": reason, "own_positions": own, "positions": len(seen)}

    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > cap:
            return result("cap")
        if deadline is not None and len(seen) % 1024 == 1 and time.monotonic() > deadline:
            return result("timeout")
        if game.is_terminal(s):
            w = game.winner(s)
            if w != player:
                return result("draw" if w is None else "loss")
            continue
        if game.current_player(s) == player:
            own += 1
            move = choose(facts, s, levels, n_levels)
            if move is None:
                return result("undefined")
            stack.append(game.step(s, move))
        else:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
    return result(None)


def map_bits(levels: dict, empty: int, level_bits: int) -> int:
    """The map's size: one bit per cell empty at its position (levelled or not) and `level_bits` per levelled cell —
    and for a second phase the same again, plus its switch as stones past the position's (1..empty)."""
    first = [c for c in levels if c >= 0]
    second = [c for c in levels if c < SWITCH]
    if second and SWITCH not in levels:
        raise ValueError("a map's second phase needs its switch")
    if max(len(first), len(second)) > empty:
        raise ValueError(f"{max(len(first), len(second))} levelled cells cannot exceed the {empty} empty cells")
    bits = empty + level_bits * len(first)
    if SWITCH in levels:
        bits += max(0, (empty - 1).bit_length()) + empty + level_bits * len(second)
    return bits


def simplify(facts: Facts, root, levels: dict, n_levels: int, cap: int = 1_000_000,
             deadline: float | None = None, verifier=None) -> dict:
    """Drop every level the rule does not need: a second phase first, whole (its switch and its levels), then each
    levelled cell of either phase in turn is cleared and kept cleared if the map still verifies (by `verifier`,
    `verify` by default). The result verifies whenever the input did; past `deadline` the remaining levels are kept."""
    check = verifier or verify
    kept = dict(levels)
    if SWITCH in kept:
        trial = {c: k for c, k in kept.items() if c >= 0}
        if check(facts, root, trial, n_levels, cap, deadline=deadline)["won"]:
            kept = trial
    for cell in sorted(c for c in kept if c != SWITCH):
        trial = {c: k for c, k in kept.items() if c != cell}
        if check(facts, root, trial, n_levels, cap, deadline=deadline)["won"]:
            kept = trial
    return kept
