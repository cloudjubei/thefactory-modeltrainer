"""Direct tests for harness.play — the watch/replay/interactive surface over a trained model.

The interactive loop is driven through injected `prompt`/`emit` callables so a scripted human can play a
full game with no real stdin, and the pure formatters (`render_replay`, `parse_human_action`, result line)
are exercised on hand-built states so a mutation to a guard changes an assertion.
"""
from __future__ import annotations

import pytest

from harness.agents import RandomAgent
from games.connect4 import Connect4
from harness.play import (
    parse_human_action,
    play_human_vs_model,
    render_replay,
    run_serve,
    serve_autoplay,
    serve_move,
)


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


def _rng():
    import random

    return random.Random(0)


def test_serve_move_starts_a_game_with_the_model_moving_first():
    game = _game()
    # human is seat 1, so the model (seat 0) opens; a fresh game has an empty history.
    res = serve_move(game, lambda g, s, r: 3, actions=[], human_seat=1, rng=_rng())
    assert res["mode"] == "move"
    assert res["model_seat"] == 0 and res["human_seat"] == 1
    assert res["actions"] == [3]
    assert len(res["frames"]) == 2
    assert res["to_move"] == 1  # back to the human
    assert res["terminal"] is False
    assert res["winner"] is None
    assert res["num_actions"] == 7
    assert set(res["legal_actions"]).issubset(set(range(7)))
    assert res["moves"] == [{"player": 0, "action": 3, "label": "col 3"}]


def test_serve_move_replays_history_then_the_model_replies():
    game = _game()
    # human (seat 0) just played col 0; the model (seat 1) should reply once, then it is the human's turn.
    res = serve_move(game, lambda g, s, r: 1, actions=[0], human_seat=0, rng=_rng())
    assert res["model_seat"] == 1
    assert res["actions"] == [0, 1]
    assert len(res["frames"]) == 3
    assert res["to_move"] == 0
    assert res["terminal"] is False


def test_serve_move_detects_a_terminal_win():
    game = _game()
    # history: model(0) col1, human(1) col0, x3 → model about to complete a vertical four in col 1.
    res = serve_move(game, lambda g, s, r: 1, actions=[1, 0, 1, 0, 1, 0], human_seat=1, rng=_rng())
    assert res["actions"] == [1, 0, 1, 0, 1, 0, 1]
    assert res["terminal"] is True
    assert res["winner"] == 0  # the model (seat 0)
    assert res["legal_actions"] == []
    assert res["to_move"] is None


def test_serve_autoplay_returns_a_full_replay():
    game = _game()
    res = serve_autoplay(game, RandomAgent(), RandomAgent(), model_seat=0, rng=_rng(), opponent_name="random")
    assert res["mode"] == "autoplay"
    assert res["model_seat"] == 0
    assert res["opponent"] == "random"
    assert res["winner"] in (0, 1, None)
    assert len(res["frames"]) == len(res["moves"]) + 1


def test_run_serve_dispatches_move_and_autoplay_and_rejects_unknown_mode():
    move = run_serve(
        {"mode": "move", "game": "connect4", "model_name": "heuristic", "actions": [], "human_seat": 0, "seed": 0}
    )
    assert move["mode"] == "move" and move["to_move"] == 0 and move["actions"] == []

    auto = run_serve(
        {"mode": "autoplay", "game": "connect4", "model_name": "heuristic", "opponent": "random", "seed": 0}
    )
    assert auto["mode"] == "autoplay" and len(auto["frames"]) >= 1

    bad = run_serve({"mode": "nope", "game": "connect4"})
    assert "error" in bad


def test_run_serve_reports_an_illegal_move_as_an_error():
    res = run_serve(
        {"mode": "move", "game": "connect4", "model_name": "heuristic", "actions": [7], "human_seat": 0, "seed": 0}
    )
    assert "error" in res
