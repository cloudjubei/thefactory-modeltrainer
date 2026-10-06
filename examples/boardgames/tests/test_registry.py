"""Direct tests for harness/registry.py — the one place a game is wired in. Worker processes rebuild a game from its
name (`resolve_game(game.name)`), so every registered name must rebuild a game that answers to that same name: a
parameterised game (Kalah's holes and counters) registered under a shared name would come back as another shape."""
from __future__ import annotations

import pytest

from harness.registry import GAMES, PERSONAS, resolve_game


@pytest.mark.parametrize("name", sorted(GAMES))
def test_every_registered_name_rebuilds_a_game_that_answers_to_it(name):
    assert resolve_game(name).name == name


def test_every_game_has_a_personas_entry():
    assert set(PERSONAS) == set(GAMES)


@pytest.mark.parametrize("name,shape", [("kalah", (6, 4)), ("kalah4x3", (4, 3)), ("kalah3x3", (3, 3))])
def test_kalah_is_registered_at_the_standard_shape_and_two_exactly_solvable_ones(name, shape):
    game = resolve_game(name)
    assert (game.holes, game.counters) == shape


def test_an_unknown_game_is_refused():
    with pytest.raises(ValueError, match="unknown game"):
        resolve_game("no_such_game")
