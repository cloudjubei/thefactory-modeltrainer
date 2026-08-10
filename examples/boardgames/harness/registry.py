"""Game + persona registry — the one place a new game (or a luck-based game's protocol personas) is wired in."""
from __future__ import annotations

from typing import Callable

from games.connect4 import Connect4
from harness.agents import Agent
from harness.game import Game

GAMES: dict[str, Callable[[], Game]] = {
    "connect4": Connect4,
}

# Per-game protocol PERSONAS (fixed-strategy archetypes) — the opponent rungs the luck-based games (Skull,
# Flip 7, Skull King) need. Connect 4 is perfect-information, so its ladder is random/heuristic/mcts only.
PERSONAS: dict[str, dict[str, Callable[[dict], Agent]]] = {
    "connect4": {},
}


def resolve_game(name: str) -> Game:
    if name not in GAMES:
        raise ValueError(f"unknown game {name!r}; known: {sorted(GAMES)}")
    return GAMES[name]()


def personas_for(name: str) -> dict[str, Callable[[dict], Agent]]:
    return PERSONAS.get(name, {})
