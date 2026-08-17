"""The game-agnostic protocol every game plug-in implements.

Deliberately OBSERVATION-based: an agent is only ever handed `observation(state, player)` — what that
player can see — so a hidden-information game (Skull, Poker, Skull King) drops into the same harness as a
perfect-information one (Connect 4). `step`/`initial_state` take an rng so stochastic games (Flip 7's deck,
a poker deal) fit too; deterministic games ignore it. The harness (self-play loop, opponent ladder, RunSummary)
never inspects a concrete state — it only calls these methods, so a new game is one file.
"""
from __future__ import annotations

import random
from typing import Any, Protocol, runtime_checkable

# A game's state is opaque to the harness — each game owns its own representation.
State = Any


@runtime_checkable
class Game(Protocol):
    name: str
    num_players: int
    # Size of the (fixed) action space, so a policy can build a legal-action mask over range(num_actions).
    num_actions: int

    def initial_state(self, rng: random.Random) -> State: ...

    def current_player(self, state: State) -> int: ...

    def legal_actions(self, state: State) -> list[int]: ...

    def step(self, state: State, action: int, rng: random.Random) -> State: ...

    def is_terminal(self, state: State) -> bool: ...

    # The winning player index, or None for a draw / an ongoing game.
    def winner(self, state: State) -> int | None: ...

    # Per-player terminal payoff (e.g. win +1 / draw 0 / loss -1, or a points vector). Length == num_players.
    def returns(self, state: State) -> list[float]: ...

    # What `player` can see — the ONLY view an agent is given (respects hidden information).
    def observation(self, state: State, player: int) -> list[float]: ...

    # A decent non-random move for the player to move — the heuristic opponent rung AND a baseline core.
    def heuristic_action(self, state: State, rng: random.Random) -> int: ...

    # A human-readable ASCII render of a state, so a sampled game can be replayed in the RunSummary / viewer.
    def render(self, state: State) -> str: ...

    # Optional: a short label for an action (defaults to str(action)) — for readable move logs.
    def action_label(self, state: State, action: int) -> str: ...

    # Optional: a hashable CANONICAL key for a position, so the MCTS transposition table collapses states
    # reached by different move orders. When a game omits it, the search falls back to the observation.
    def state_key(self, state: State) -> object: ...


@runtime_checkable
class SolvableGame(Game, Protocol):
    """A Game that also exposes the per-game hooks the optimal-play book engine (harness/book.py) needs. Only
    these four are game-specific; the tablebase, incremental builder, early-termination, symmetry augmentation
    and BookAgent are all game-agnostic and consume this interface. A new solvable game is one file:
      - `ply(state)`          — pieces placed (the book builder orders deepest/cheapest-first by it);
      - `canonical_key(state)`— a symmetry-reduced integer key (a position and its symmetric images collapse);
      - `position_value(state, book)` — exact value to the player to move (win +1 / draw 0 / loss -1), the
        solver, which may read `book` for already-solved children (the bottom-up wall-break);
      - `symmetries()`        — the exploitable symmetries as action (source→dest) permutations, for net
        augmentation; identity-only when a game has none (the rest still works, just no space saving)."""

    def ply(self, state: State) -> int: ...

    def canonical_key(self, state: State) -> int: ...

    def position_value(self, state: State, book: Any = None) -> int: ...

    def symmetries(self) -> list[list[int]]: ...

