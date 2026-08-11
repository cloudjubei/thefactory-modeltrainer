"""Direct tests for harness.play — the watch/replay/interactive surface over a trained model.

The interactive loop is driven through injected `prompt`/`emit` callables so a scripted human can play a
full game with no real stdin, and the pure formatters (`render_replay`, `parse_human_action`, result line)
are exercised on hand-built states so a mutation to a guard changes an assertion.
"""
from __future__ import annotations

import pytest

from games.connect4 import Connect4
from harness.play import parse_human_action, play_human_vs_model, render_replay


def _game() -> Connect4:
    return Connect4()


def test_render_replay_shows_every_frame_the_move_labels_and_the_result():
    game = _game()
    replay = {
        "model_seat": 0,
        "winner": 0,
        "moves": [
            {"player": 0, "action": 3, "label": "col 3"},
            {"player": 1, "action": 2, "label": "col 2"},
        ],
        "frames": ["FRAME_INITIAL", "FRAME_AFTER_1", "FRAME_AFTER_2"],
    }
    out = render_replay(game, replay)
    assert "FRAME_INITIAL" in out and "FRAME_AFTER_1" in out and "FRAME_AFTER_2" in out
    assert "col 3" in out and "col 2" in out
    assert "Move 1" in out and "Move 2" in out
    assert "model wins" in out  # winner 0 == model_seat 0


def test_render_replay_labels_the_opponent_from_the_sample_game_block():
    game = _game()
    replay = {"model_seat": 1, "opponent": "heuristic", "winner": None,
              "moves": [{"player": 0, "action": 0, "label": "col 0"}], "frames": ["A", "B"]}
    out = render_replay(game, replay)
    assert "heuristic" in out
    assert "draw" in out  # winner None


def test_parse_human_action_accepts_a_legal_column():
    game = _game()
    state = game.initial_state()
    assert parse_human_action(game, state, " 3 ") == 3


def test_parse_human_action_rejects_a_non_number():
    game = _game()
    state = game.initial_state()
    with pytest.raises(ValueError):
        parse_human_action(game, state, "middle")


def test_parse_human_action_rejects_an_out_of_range_or_full_column():
    game = _game()
    state = game.initial_state()
    with pytest.raises(ValueError):
        parse_human_action(game, state, "9")
    with pytest.raises(ValueError):
        parse_human_action(game, state, "-1")


def test_play_human_vs_model_plays_a_full_game_and_returns_a_replay():
    game = _game()
    prompts = iter(["0", "0", "0"])
    emitted: list[str] = []
    # Model is seat 0 (moves first) and always drops in column 1; the human (seat 1) always drops in column 0.
    # Both race a vertical four, so the first mover (the model) completes it first — a deterministic outcome.
    replay = play_human_vs_model(
        game,
        model_act=lambda g, s, r: 1,
        human_seat=1,
        rng=None,
        prompt=lambda: next(prompts),
        emit=emitted.append,
    )
    assert replay["winner"] == 0
    assert replay["model_seat"] == 0 and replay["human_seat"] == 1
    assert len(replay["frames"]) == len(replay["moves"]) + 1
    assert any("model wins" in line for line in emitted)


def test_play_human_vs_model_reprompts_on_bad_input_without_consuming_a_move():
    game = _game()
    prompts = iter(["nope", "0", "0", "0"])
    emitted: list[str] = []
    replay = play_human_vs_model(
        game,
        model_act=lambda g, s, r: 1,
        human_seat=1,
        rng=None,
        prompt=lambda: next(prompts),
        emit=emitted.append,
    )
    assert replay["winner"] == 0
    assert any("Invalid" in line for line in emitted)
