"""Game + persona registry — the one place a new game (or a luck-based game's protocol personas) is wired in."""
from __future__ import annotations

from functools import partial
from typing import Callable

from games.connect4 import Connect4
from games.checkers import Checkers
from games.kalah import Kalah
from games.othello import Othello
from games.ninemensmorris import NineMensMorris
from games.tictactoe import TicTacToe
from harness.agents import Agent
from harness.game import Game

GAMES: dict[str, Callable[[], Game]] = {
    "connect4": Connect4,
    "othello": Othello,
    "checkers": Checkers,
    "tictactoe": TicTacToe,
    "ninemensmorris": NineMensMorris,
    "kalah": Kalah,
    "kalah4x3": partial(Kalah, 4, 3),
    "kalah3x3": partial(Kalah, 3, 3),
}

# Per-game protocol PERSONAS (fixed-strategy archetypes) — the opponent rungs the luck-based games (Skull,
# Flip 7, Skull King) need. Perfect-information games use random/heuristic/mcts/book only.
PERSONAS: dict[str, dict[str, Callable[[dict], Agent]]] = {
    "connect4": {},
    "othello": {},
    "checkers": {},
    "tictactoe": {},
    "ninemensmorris": {},
    "kalah": {},
    "kalah4x3": {},
    "kalah3x3": {},
}


def resolve_game(name: str) -> Game:
    if name not in GAMES:
        raise ValueError(f"unknown game {name!r}; known: {sorted(GAMES)}")
    return GAMES[name]()


def personas_for(name: str) -> dict[str, Callable[[dict], Agent]]:
    return PERSONAS.get(name, {})
