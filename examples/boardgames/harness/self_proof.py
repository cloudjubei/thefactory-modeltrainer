"""§3.7 — proving won positions with no solver. The §3.6 builder asked an exact solver which moves win: to choose a
move node's move, and to score a map by the exceptions it needs. Here nothing is asked. `failing` has
harness.steady_exceptions.needed's contract without its oracle: it walks every line a map plays and charges each of the
side's positions whose move ended a line unwon, so harness.steady_local.local_search can score a map by the lines it
loses (`map_search`). `Prover` proves a position for the side to move there the only two ways the rules allow: a map
that wins every line from it (harness.steady_state.verify, a proof by itself), or a safe move whose every reply is
already proved. A position whose every safe move fails is refuted; one the budget cut short is neither, and nothing is
remembered about it. With `search_below`, a map is searched for only where the empty map's walk covers at most that
many of the side's positions; a larger position is split into a move, so proofs are built from the bottom up where
a map's walk is cheap. With `compose`, every walk stops at positions already proved, so a map need only win the lines
no deeper proof wins. What is proved is a strategy in harness.strategy_file's shape, so it can be replayed and written
as a file, or with `compose` a composed strategy (harness.composed), checked by its own walk. A solver may grade the
result afterwards; it is never a means."""
from __future__ import annotations

import time

from harness.steady_local import local_search
from harness.steady_state import Facts, choose, restrict, simplify, verify

CHECK_EVERY = 1024
REPLIES_LIMIT = 1_000_000
_OWN = object()


class Replies:
    """The side's next positions after a move at one of its positions, each with its key, remembered: they depend on
    the position and the move, never on the map, so the trials of a search, which change one cell at a time and walk
    nearly the same lines, step the game once for each. Emptied when it would pass `limit` entries."""

    def __init__(self, game, limit: int = REPLIES_LIMIT):
        self.game, self.limit = game, limit
        self.memo: dict = {}
        self.cleared = 0

    def __call__(self, state, key, move):
        entry = (key, move)
        if entry not in self.memo:
            if len(self.memo) >= self.limit:
                self.memo.clear()
                self.cleared += 1
            self.memo[entry] = _keyed(self.game, state, move)
        return self.memo[entry]


def _keyed(game, state, move):
    out = _replies(game, game.current_player(state), game.step(state, move))
    return None if out is None else [(c, game.state_key(c)) for c in out]


def failing(facts: Facts, root, levels: dict, n_levels: int, _winning, cap: int, deadline: float | None = None,
            limit: int | None = None, replies: Replies | None = None, trace: bool = False, stop=None) -> dict:
    """{"status": "ok" | "cap" | "timeout" | "over", "exceptions": {position key: None}, "own_positions"} — one walk
    over every line the map plays from `root`. A position of the side is charged when the map gives it no move, or its
    move ends the game unwon, or a reply to its move does. The walk goes on past a charged position's other lines, and
    stops as "over" once more than `limit` positions are charged: a search that would refuse the map need not finish
    its walk. With `replies`, each move's replies are remembered across walks. With `trace`, the result also holds
    "came": {key: (position, the key of the side's position before it on the walk, None at the root)}, so the line to
    each charged position can be read back. A position in `stop` other than the root is not walked past: it is
    already proved, so a composed strategy (harness.composed) hands play to its own proof there."""
    game = facts.game
    charged: dict = {}
    seen: set = set()
    came: dict = {}
    own = 0
    stack = [(root, game.state_key(root), None)]
    while stack:
        s, key, parent = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        if trace:
            came[key] = (s, parent)
        if stop is not None and parent is not None and key in stop:
            continue
        if len(seen) > cap:
            return {"status": "cap", "exceptions": {}, "own_positions": own}
        if deadline is not None and len(seen) % CHECK_EVERY == 1 % CHECK_EVERY and time.monotonic() > deadline:
            return {"status": "timeout", "exceptions": {}, "own_positions": own}
        own += 1
        move = choose(facts, s, levels, n_levels)
        nexts = None if move is None else _keyed(game, s, move) if replies is None else replies(s, key, move)
        if nexts is None:
            charged[key] = None
            if limit is not None and len(charged) > limit:
                return {"status": "over", "exceptions": {}, "own_positions": own}
        else:
            stack.extend((c, k, key) for c, k in nexts)
    out = {"status": "ok", "exceptions": charged, "own_positions": own}
    return {**out, "came": came} if trace else out


def _replies(game, player, child):
    """The side's next positions after its move led to `child`; [] when every line is won at once; None when the
    move, or some reply to it, ends the game unwon."""
    if game.is_terminal(child):
        return [] if game.winner(child) == player else None
    if game.current_player(child) == player:
        return [child]
    out = []
    for b in game.legal_actions(child):
        c = game.step(child, b)
        if not game.is_terminal(c):
            out.append(c)
        elif game.winner(c) != player:
            return None
    return out


def map_search(facts: Facts, n_levels: int, level_bits: int, rng, seconds: float, cap: int):
    """A `Prover` search: harness.steady_local's hill-climb with `failing` as its walk and no oracle."""
    def search(state, start):
        return local_search(facts, state, n_levels, level_bits, None, rng, seconds, cap, start=start, walker=failing)
    return search


def _random_trial(rng, cells: list, n_levels: int, levels: dict) -> dict:
    cell, level = rng.choice(cells), rng.randrange(n_levels + 1)
    trial = dict(levels)
    if level == n_levels:
        trial.pop(cell, None)
    else:
        trial[cell] = level
    return trial


def aimed_trial(facts: Facts, n_levels: int, rng, levels: dict, walk: dict):
    """`levels` changed so that, at one of the side's positions on the line to one of `walk`'s charged positions (a
    traced `failing` walk), the rule plays another of the moves it chooses among there: that move's cell gets a level
    below the one that decides there, or, where that is already the first, the map's own move's cell goes one level
    up. None when no position on the line has a choice (an immediate win or a single move)."""
    key = rng.choice(list(walk["exceptions"]))
    line = []
    while key is not None:
        state, key = walk["came"][key]
        line.append(state)
    places = [s for s in line if not facts.of(s)[0] and len(facts.of(s)[1]) > 1]
    if not places:
        return None
    s = rng.choice(places)
    _, moves, cell = facts.of(s)
    mine = choose(facts, s, levels, n_levels)
    other = rng.choice([m for m in moves if m != mine])
    top = min(levels.get(cell[m], n_levels) for m in moves)
    trial = dict(levels)
    if top > 0:
        trial[cell[other]] = rng.randrange(min(top, n_levels))
    else:
        trial[cell[other]] = 0
        trial[cell[mine]] = 1
    return trial


def _climb(facts: Facts, n_levels: int, cap: int, seconds, evaluations, memo, propose, trace: bool):
    def search(state, start, stop=None):
        deadline = None if seconds is None else time.monotonic() + seconds
        cells = facts.empty_cells(state)
        current = dict(start or {})
        now = failing(facts, state, current, n_levels, None, cap, deadline, replies=memo, trace=trace, stop=stop)
        if now["status"] != "ok":
            return None
        walks = 1
        while (now["exceptions"] and (evaluations is None or walks < evaluations)
               and (deadline is None or time.monotonic() < deadline)):
            trial = propose(cells, current, now)
            if trial == current:
                continue
            walks += 1
            r = failing(facts, state, trial, n_levels, None, cap, deadline, limit=len(now["exceptions"]), replies=memo,
                        trace=trace, stop=stop)
            if r["status"] == "ok":
                current, now = trial, r
        return {"status": "found" if not now["exceptions"] else "budget", "levels": current,
                "exceptions": now["exceptions"], "own_positions": now["own_positions"], "evaluations": walks}
    return search


def bounded_search(facts: Facts, n_levels: int, rng, cap: int, seconds: float | None = None,
                   evaluations: int | None = None, replies=_OWN):
    """A `Prover` search that hill-climbs on the count of failing positions alone: one cell's level is changed or
    cleared at a time, and the change is kept unless the map fails at more positions. Each trial's walk is cut short
    once it fails at more positions than the map in hand, which a full walk would only confirm. It ends at a map with
    no failing position, after `seconds`, or after `evaluations` walks; None when the first walk cannot be finished.
    Its walks share `replies` (by default its own `Replies`, kept across its searches; None for none)."""
    memo = Replies(facts.game) if replies is _OWN else replies
    return _climb(facts, n_levels, cap, seconds, evaluations, memo,
                  lambda cells, current, now: _random_trial(rng, cells, n_levels, current), False)


def aimed_search(facts: Facts, n_levels: int, rng, cap: int, seconds: float | None = None,
                 evaluations: int | None = None, replies=_OWN, share: float = 1.0):
    """`bounded_search` with `share` of its trials aimed (`aimed_trial`) at the lines that failed; the rest, and any
    aimed trial that finds no position with a choice on its line, change a random cell's level."""
    memo = Replies(facts.game) if replies is _OWN else replies

    def propose(cells, current, now):
        trial = aimed_trial(facts, n_levels, rng, current, now) if share >= 1 or rng.random() < share else None
        return _random_trial(rng, cells, n_levels, current) if trial is None else trial
    return _climb(facts, n_levels, cap, seconds, evaluations, memo, propose, True)


class Prover:
    def __init__(self, facts: Facts, n_levels: int, cap: int, search=None, reuse_window: int = 0,
                 max_calls: int | None = None, deadline: float | None = None, search_below: int | None = None,
                 compose: bool = False):
        self.facts, self.game, self.n_levels, self.cap = facts, facts.game, n_levels, cap
        self.search, self.reuse_window, self.max_calls, self.deadline = search, reuse_window, max_calls, deadline
        self.search_below, self.compose = search_below, compose
        self.maps: list = [{}]
        self.proved: dict = {}
        self.refuted: set = set()
        self.calls = 0

    def prove(self, state):
        """True when `state` is proved won for the side to move there, False when it is refuted, None when the budget
        ran out first."""
        key = self.game.state_key(state)
        if key in self.proved:
            return True
        if key in self.refuted:
            return False
        if self._spent():
            return None
        self.calls += 1
        first = self._leaf(state, key)
        if first is True:
            return True
        safe = self.facts.safe_moves(state)
        unknown = False
        for action in sorted(safe, key=lambda a: (a != first, a)):
            result = self._move(state, action)
            if result:
                self.proved[key] = {"move": action}
                return True
            unknown = unknown or result is None
        if unknown:
            return None
        self.refuted.add(key)
        return False

    def _spent(self) -> bool:
        return ((self.max_calls is not None and self.calls >= self.max_calls)
                or (self.deadline is not None and time.monotonic() > self.deadline))

    def _leaf(self, state, key):
        """True when a map proves `state`; else the move the searched map would play, to try first."""
        empty = self.facts.empty_cells(state)
        recent = range(len(self.maps) - 1, max(0, len(self.maps) - 1 - self.reuse_window), -1)
        for i in [0, *recent]:
            if self._wins(state, self.maps[i]):
                self.proved[key] = {"leaf": i}
                return True
        if self.search is None or not self._small(state):
            return None
        start = restrict(self.maps[-1], empty)
        found = self.search(state, start, stop=self.proved) if self.compose else self.search(state, start)
        if found is None:
            return None
        if not found["exceptions"] and self._wins(state, found["levels"]):
            kept = simplify(self.facts, state, found["levels"], self.n_levels, self.cap, deadline=self.deadline,
                            verifier=self._verifier if self.compose else None)
            self.maps.append(kept)
            self.proved[key] = {"leaf": len(self.maps) - 1}
            return True
        return choose(self.facts, state, found["levels"], self.n_levels)

    def _small(self, state) -> bool:
        if self.search_below is None:
            return True
        walk = failing(self.facts, state, {}, self.n_levels, None, self.search_below, self.deadline,
                       stop=self.proved if self.compose else None)
        return walk["status"] == "ok"

    def _wins(self, state, levels: dict) -> bool:
        if not self.compose:
            return verify(self.facts, state, levels, self.n_levels, self.cap, deadline=self.deadline)["won"]
        walk = failing(self.facts, state, levels, self.n_levels, None, self.cap, self.deadline, limit=0,
                       stop=self.proved)
        return walk["status"] == "ok"

    def _verifier(self, _facts, state, levels, _n_levels, _cap, deadline=None) -> dict:
        return {"won": self._wins(state, levels)}

    def _move(self, state, action):
        replies = _replies(self.game, self.game.current_player(state), self.game.step(state, action))
        if replies is None:
            return False
        unknown = False
        for reply in replies:
            result = self.prove(reply)
            if result is False:
                return False
            unknown = unknown or result is None
        return None if unknown else True

    def strategy(self, root) -> dict:
        """The proved strategy from `root`: {"nodes", "maps"} as harness.strategy_file reads one, or with `compose` as
        harness.composed reads one, holding every proved position the play reaches."""
        if self.compose:
            return self._composed(root)
        game = self.game
        player = game.current_player(root)
        nodes: dict = {}
        stack = [root]
        while stack:
            s = stack.pop()
            key = game.state_key(s)
            if key in nodes:
                continue
            nodes[key] = self.proved[key]
            if "move" in nodes[key]:
                stack.extend(_replies(game, player, game.step(s, nodes[key]["move"])))
        return {"nodes": nodes, "maps": self.maps}

    def _composed(self, root) -> dict:
        game = self.game
        player = game.current_player(root)
        nodes: dict = {}
        seen: set = set()
        stack = [(root, None)]
        while stack:
            s, current = stack.pop()
            key = game.state_key(s)
            if (key, current) in seen or game.is_terminal(s):
                continue
            seen.add((key, current))
            if game.current_player(s) != player:
                stack.extend((game.step(s, b), current) for b in game.legal_actions(s))
                continue
            if key in self.proved:
                nodes[key] = node = self.proved[key]
                current = node.get("leaf")
                move = node["move"] if "move" in node else choose(self.facts, s, self.maps[current], self.n_levels)
            else:
                move = choose(self.facts, s, self.maps[current], self.n_levels)
            stack.append((game.step(s, move), current))
        return {"nodes": nodes, "maps": self.maps}

