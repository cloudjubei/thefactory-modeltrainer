"""The process's generic value steps assume the side to move alternates: a child's value is the parent mover's
negated. Kalah breaks that — a last counter in the mover's store moves again — so each step below misreads an extra
move. Strict xfails: each passes, and so fails the suite, the moment its step compares movers instead of negating
(as the search backup already does, harness.neural `v if mover == leaf_player else -v`)."""
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


@pytest.mark.xfail(strict=True, reason="child_move_value negates a proven child whose mover is the parent's mover")
def test_a_proven_child_after_an_extra_move_keeps_its_value_for_the_same_mover():
    game, s, child = _extra_move_position()
    assert child_move_value(game, s, 1, {state_key(game, child): 1}) == 1


@pytest.mark.xfail(strict=True, reason="certify negates every child's value, so an optimal extra move reads as losing")
def test_exact_play_certifies_through_extra_moves():
    game = Kalah(2, 1)
    root = game.initial_state()
    r = certify(game, root, 0, lambda states: [game.optimal_actions(s)[0] for s in states], game.position_value)
    assert r["certified"] and r["complete_game"]
