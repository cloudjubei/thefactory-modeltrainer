"""Steady states (§3.6 S1): a compact rule that plays the rest of a placement game from one position, for the side to
move there, after WeakC4 (2swap; https://2swap.github.io/WeakC4/explanation/). The rule is a priority map — a level
0..levels-1 on some of the cells empty at that position — read the same way at every later position:

  1. play a move that wins at once (the lowest such action);
  2. keep the SAFE moves — those after which the opponent cannot win at once; if exactly one is safe, play it;
  3. otherwise go through the levels in order and play the safe move whose cell is at the first level holding
     exactly one safe move; a level holding none or several is passed over;
  4. otherwise the rule has no move: it is undefined there.

A move's cell is read from the game's own observation (the one board cell it fills), so the language knows nothing
about any particular game; it applies to placement games, where every move fills exactly one cell.

`verify` walks every position the rule reaches against every reply and accepts only if every line ends in a win for
the side the rule plays — an undefined position, a draw or a loss rejects it. It never consults a solver. Facts that
do not depend on the map (immediate wins, safe moves, their cells) are cached per position across calls, in a cache
emptied whenever it reaches its limit (~1 KB a position: a 4-hour build reached 9.6 GB a worker unbounded)."""
from __future__ import annotations


class Facts:
    """Per-position facts for one game, independent of any priority map: immediate wins, safe moves, each safe
    move's cell. Shared by the verifier and the search so a position is analysed once (until the cache is emptied)."""

    def __init__(self, game, limit: int = 2_000_000):
        self.game = game
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

    def of(self, state) -> tuple:
        """(winning actions, safe actions, {safe action: its cell}) at `state`."""
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
            if len(self._facts) >= self.limit:
                self._facts.clear()
                self.cleared += 1
            self._facts[key] = (wins, safe, {a: self.placed_cell(state, a) for a in safe})
        return self._facts[key]


def choose(facts: Facts, state, levels: dict, n_levels: int):
    """The rule's move at `state` under the priority map `levels` ({cell: level}), or None where it is undefined."""
    wins, safe, cell = facts.of(state)
    if wins:
        return min(wins)
    if len(safe) == 1:
        return safe[0]
    for k in range(n_levels):
        at = [a for a in safe if levels.get(cell[a]) == k]
        if len(at) == 1:
            return at[0]
    return None


def verify(facts: Facts, root, levels: dict, n_levels: int, cap: int = 1_000_000) -> dict:
    """Whether the rule wins every line from `root` for the side to move there. Returns {"won", "reason" (None, or
    the first failure: "undefined", "draw", "loss", "cap"), "own_positions" (the rule's decision positions walked),
    "positions" (all positions walked)}."""
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
    """The map's size: one bit per cell empty at its position (levelled or not) and `level_bits` per levelled cell."""
    if len(levels) > empty:
        raise ValueError(f"{len(levels)} levelled cells cannot exceed the {empty} empty cells")
    return empty + level_bits * len(levels)


def simplify(facts: Facts, root, levels: dict, n_levels: int, cap: int = 1_000_000) -> dict:
    """Drop every level the rule does not need: each levelled cell in turn is cleared and kept cleared if the map
    still verifies. The result verifies whenever the input did."""
    kept = dict(levels)
    for cell in sorted(levels):
        trial = {c: k for c, k in kept.items() if c != cell}
        if verify(facts, root, trial, n_levels, cap)["won"]:
            kept = trial
    return kept
