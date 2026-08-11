"""Dependency-light agents — the first `model_name` cores AND the opponent-ladder rungs (no torch).

An Agent is any object with `act(game, state, rng) -> int`. The self-play harness hands agents the full
state (search needs it); a live-deployable LEARNED policy must instead rely only on `game.observation(...)`
(see `select_action` on a policy), which is what a BoardGameArena bridge would call.

The opponent axis is an extensible REGISTRY, not a fixed list: baseline rungs (`random`/`heuristic`/`mcts`)
live here; a game registers its protocol PERSONAS; and champion CHECKPOINTS are added as they are crowned —
the league/population an AlphaGo-style trainer plays against. `resolve_agent` builds an agent from a name.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Protocol

from harness.game import Game, State


class Agent(Protocol):
    def act(self, game: Game, state: State, rng: random.Random) -> int: ...


class RandomAgent:
    kind = "random"

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        return rng.choice(game.legal_actions(state))


class HeuristicAgent:
    kind = "heuristic"

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        return game.heuristic_action(state, rng)


class _Node:
    """One search-tree node = one game POSITION (keyed in the transposition table). Holds per-child edge
    statistics; `select` is UCT over the children."""

    __slots__ = ("legal", "child_n", "child_w", "total")

    def __init__(self, legal: list[int]):
        self.legal = legal
        self.child_n = {a: 0 for a in legal}
        self.child_w = {a: 0.0 for a in legal}
        self.total = 0

    def select(self, c_puct: float) -> int:
        log_total = math.log(self.total + 1.0)
        best_a, best_u = self.legal[0], -math.inf
        for a in self.legal:
            n = self.child_n[a]
            if n == 0:
                return a  # try every child once before exploiting
            u = self.child_w[a] / n + c_puct * math.sqrt(log_total / n)
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    def update(self, action: int, value: float) -> None:
        self.child_n[action] += 1
        self.child_w[action] += value
        self.total += 1


def state_key(game: Game, state: State) -> object:
    """A hashable, canonical key for a position — a game's own `state_key` when it provides one (so
    transpositions collapse), else a fallback from the current player's observation + side to move."""
    fn = getattr(game, "state_key", None)
    if fn is not None:
        return fn(state)
    return (tuple(game.observation(state, game.current_player(state))), game.current_player(state))


class MctsAgent:
    """Generic UCT Monte-Carlo Tree Search over the game protocol — a REAL tree, strength scales with `sims`.

    Each simulation SELECTS down the tree by UCT to a leaf, EXPANDS it, random-ROLLS OUT to a terminal, and
    BACKS UP the outcome along the visited edges (each from the perspective of the player who moved). Nodes
    are keyed by POSITION in a transposition table (`_tree`), so lines that reach the same board share their
    statistics AND the table persists across the agent's moves within a game — the search keeps what it has
    learned about mid-game positions instead of re-deriving it each turn. Domain-oblivious: only the Game
    protocol is used, so the same search plays every game. Reports `sims_used` for the harness cost tally.
    """

    kind = "mcts"

    def __init__(self, sims: int = 80, c_puct: float = 1.4):
        self.sims = max(1, int(sims))
        self.c_puct = c_puct
        self.sims_used = 0
        self._tree: dict[object, _Node] = {}

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        legal = game.legal_actions(state)
        if len(legal) == 1:
            self.sims_used += 1
            return legal[0]
        root_key = state_key(game, state)
        for _ in range(self.sims):
            self.sims_used += 1
            self._simulate(game, state, root_key, rng)
        root = self._tree[root_key]
        # robust child: the most-visited action, ties broken by mean value
        return max(legal, key=lambda a: (root.child_n[a], root.child_w[a] / root.child_n[a] if root.child_n[a] else 0.0))

    def _simulate(self, game: Game, state: State, root_key: object, rng: random.Random) -> None:
        path: list[tuple[_Node, int, int]] = []  # (node, action, mover) edges walked this simulation
        s = state
        key = root_key
        while True:
            if game.is_terminal(s):
                returns = game.returns(s)
                break
            node = self._tree.get(key)
            if node is None:  # EXPAND this leaf, then evaluate it with a random rollout
                self._tree[key] = _Node(game.legal_actions(s))
                returns = self._rollout(game, s, rng)
                break
            mover = game.current_player(s)
            action = node.select(self.c_puct)
            path.append((node, action, mover))
            s = game.step(s, action, rng)
            key = state_key(game, s)
        for node, action, mover in path:
            node.update(action, returns[mover])

    def _rollout(self, game: Game, state: State, rng: random.Random) -> list[float]:
        while not game.is_terminal(state):
            state = game.step(state, rng.choice(game.legal_actions(state)), rng)
        return game.returns(state)


# --- the opponent / core registry -------------------------------------------------------------------------

_BASELINE: dict[str, Callable[[dict], Agent]] = {
    "random": lambda cfg: RandomAgent(),
    "heuristic": lambda cfg: HeuristicAgent(),
    "mcts": lambda cfg: MctsAgent(sims=int(cfg.get("mcts_sims", 80))),
}


def resolve_agent(name: str, cfg: dict, personas: dict[str, Callable[[dict], Agent]] | None = None) -> Agent:
    """Build an agent by NAME from the baseline cores, a game's registered personas, or (later) a champion
    checkpoint (`champion:<ref>`). Unknown names fail loudly so a mistyped opponent can't silently pass."""
    if name in _BASELINE:
        return _BASELINE[name](cfg)
    if name == "alphazero":
        # Lazy import: the neural core needs torch, but the light cores must not, so importing it is deferred
        # to the moment an alphazero agent is actually requested.
        from harness.neural import build_alphazero_agent

        return build_alphazero_agent(cfg)
    if personas and name in personas:
        return personas[name](cfg)
    if name.startswith("champion:"):
        raise NotImplementedError("champion checkpoints load once league play lands")
    raise ValueError(f"unknown agent/opponent {name!r}; known: {sorted(_BASELINE)} + ['alphazero'] + personas {sorted(personas or {})}")
