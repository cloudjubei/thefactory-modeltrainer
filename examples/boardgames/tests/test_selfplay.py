import random

from games.connect4 import Connect4
from harness.selfplay import evaluate_vs_opponent, play_match
from harness.agents import MctsAgent, RandomAgent


def test_play_match_returns_a_decisive_or_drawn_result_with_a_replay():
    game = Connect4()
    res = play_match(game, MctsAgent(sims=40), RandomAgent(), model_seat=0, rng=random.Random(3), capture=True)
    assert res.winner in (0, 1, None)
    assert res.moves >= 7  # at least 4+ pieces each side to reach a terminal
    assert res.replay is not None
    assert len(res.replay["frames"]) == res.moves + 1  # initial frame + one per move
    assert res.sims_used > 0


def test_evaluate_folds_the_C_metric_battery_and_a_sample_game():
    game = Connect4()
    ev = evaluate_vs_opponent(game, "mcts", "random", {"mcts_sims": 50}, n_games=20, rng=random.Random(5))
    assert ev.games == 20
    assert ev.win_rate >= 0.7  # mcts dominates random
    assert 0.0 <= ev.draw_rate <= 1.0
    assert abs(ev.win_rate + ev.draw_rate + ev.loss_rate - 1.0) < 1e-9
    assert 0.0 <= ev.first_player_seat_winrate <= 1.0
    assert len(ev.win_rate_series) == 20
    assert ev.sims_used > 0
    assert ev.sample_game is not None and ev.sample_game["opponent"] == "random"


def test_mirror_matchup_is_roughly_balanced_and_flags_nothing():
    game = Connect4()
    ev = evaluate_vs_opponent(game, "random", "random", {}, n_games=30, rng=random.Random(9))
    # random-vs-random: neither side has an edge beyond the first-move advantage; win_rate near 0.5.
    assert 0.2 <= ev.win_rate <= 0.8
    assert "all_draws" not in ev.flags
