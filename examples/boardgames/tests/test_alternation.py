"""The process's generic value steps must not assume the side to move alternates: a child's value is the parent
mover's only when the same player moves next, and its negation otherwise. Kalah breaks alternation — a last counter in
the mover's store moves again — so each step is checked on an extra move (the search backup already compared movers,
harness.neural `v if mover == leaf_player else -v`)."""
from __future__ import annotations

import pytest

from games.kalah import Kalah, KalahState
from harness.agents import child_move_value, state_key
from harness.certify import certify


def _extra_move_position():
    game = Kalah(2, 1)
    s = KalahState(pits=(1, 1, 1, 1), stores=(0, 0), to_move=0, done=False)
    child = game.step(s, 1)
    return game, s, child


def test_the_fixture_is_an_extra_move_into_a_won_position():
    game, s, child = _extra_move_position()
    assert child.to_move == s.to_move and not game.is_terminal(child) and game.position_value(child) == 1


def test_a_proven_child_after_an_extra_move_keeps_its_value_for_the_same_mover():
    game, s, child = _extra_move_position()
    assert child_move_value(game, s, 1, {state_key(game, child): 1}) == 1


def test_exact_play_certifies_through_extra_moves():
    game = Kalah(2, 1)
    root = game.initial_state()
    r = certify(game, root, 0, lambda states: [game.optimal_actions(s)[0] for s in states], game.position_value)
    assert r["certified"] and r["complete_game"]


def test_a_proven_child_after_a_normal_move_is_still_negated():
    game = Kalah(2, 1)
    s = KalahState(pits=(1, 1, 1, 1), stores=(0, 0), to_move=0, done=False)
    child = game.step(s, 0)
    assert child.to_move != s.to_move
    assert child_move_value(game, s, 0, {state_key(game, child): 1}) == -1


def test_a_losing_extra_move_is_caught_by_the_certificate():
    game = Kalah(2, 1)
    root = game.initial_state()
    r = certify(game, root, 0, lambda states: [min(game.legal_actions(s), key=lambda a: game.move_margin(s, a))
                                               for s in states], game.position_value)
    assert not r["certified"] and r["failures"] >= 1


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 3)])
def test_exact_play_certifies_whole_games_that_mix_extra_and_normal_moves(shape):
    game = Kalah(*shape)
    root = game.initial_state()
    r = certify(game, root, 0, lambda states: [game.optimal_actions(s)[0] for s in states], game.position_value)
    assert r["certified"] and r["complete_game"] and r["root_value"] == game.position_value(root)
