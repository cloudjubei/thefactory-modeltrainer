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


class MctsAgent:
    """Generic UCT Monte-Carlo Tree Search over the game protocol — strength scales with `sims`.

    Uses random rollouts to a terminal, backing up each node from the perspective of the player who moved
    into it. Domain-oblivious: only `current_player`/`legal_actions`/`step`/`is_terminal`/`returns` are used,
    so the same search plays every game. `sims` counts simulations per move — the compute knob the §C gates
    sweep over. Reports `sims_used` so the harness can total the compute a run spent.
    """

    kind = "mcts"

    def __init__(self, sims: int = 80, c_puct: float = 1.4):
        self.sims = max(1, int(sims))
        self.c_puct = c_puct
        self.sims_used = 0

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        legal = game.legal_actions(state)
        if len(legal) == 1:
            self.sims_used += 1
            return legal[0]
        root_player = game.current_player(state)
        # child stats keyed by action: [visits, total_value_for_root_player]
        stats: dict[int, list[float]] = {a: [0.0, 0.0] for a in legal}
        for _ in range(self.sims):
            self.sims_used += 1
            action = self._select(stats, sum(s[0] for s in stats.values()))
            child = game.step(state, action, rng)
            value = self._rollout(game, child, root_player, rng)
            stats[action][0] += 1
            stats[action][1] += value
        # pick the most-visited action (robust child), ties broken by mean value
        best = max(legal, key=lambda a: (stats[a][0], stats[a][1] / stats[a][0] if stats[a][0] else 0.0))
        return best

    def _select(self, stats: dict[int, list[float]], total: float) -> int:
        log_total = math.log(total + 1.0)
        best_a, best_u = None, -math.inf
        for a, (visits, value) in stats.items():
            if visits == 0:
                return a
            exploit = value / visits
            explore = self.c_puct * math.sqrt(log_total / visits)
            u = exploit + explore
            if u > best_u:
                best_u, best_a = u, a
        return best_a  # type: ignore[return-value]

    def _rollout(self, game: Game, state: State, root_player: int, rng: random.Random) -> float:
        while not game.is_terminal(state):
            state = game.step(state, rng.choice(game.legal_actions(state)), rng)
        return game.returns(state)[root_player]


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
    if personas and name in personas:
        return personas[name](cfg)
    if name.startswith("champion:"):
        raise NotImplementedError("champion checkpoints load once the neural core lands (league play)")
    raise ValueError(f"unknown agent/opponent {name!r}; known: {sorted(_BASELINE)} + personas {sorted(personas or {})}")
