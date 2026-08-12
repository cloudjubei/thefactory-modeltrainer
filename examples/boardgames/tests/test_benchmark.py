"""The distance-to-optimal benchmark (harness/benchmark.py): the oracle scores a perfect rate, a weaker policy
does not — so `oracle_optimality_rate` is a real, monotone 'how close to solved' gauge. Corpus is late-game
(high `min_moves`) so the exact solves stay fast."""
import random

from games.connect4 import Connect4
from harness.benchmark import evaluate_optimality, optimality_rate, sample_solvable_positions
from harness.solver import OracleAgent


def test_oracle_is_optimal_in_every_position():
    game = Connect4()
    rng = random.Random(0)
    oracle = OracleAgent()
    states = sample_solvable_positions(game, n=15, min_moves=26, seed=1)
    assert states, "expected some sampled positions"
    optimal, total = optimality_rate(states, lambda s: oracle.act(game, s, rng))
    assert optimal == total  # perfect play is optimal everywhere → rate 1.0


def test_a_weaker_policy_scores_below_the_oracle():
    game = Connect4()
    rng = random.Random(0)
    states = sample_solvable_positions(game, n=40, min_moves=24, seed=2)
    oracle = OracleAgent()
    res_random = evaluate_optimality(game, lambda s: rng.choice(game.legal_actions(s)), states=states)
    res_oracle = evaluate_optimality(game, lambda s: oracle.act(game, s, rng), states=states)
    assert res_oracle["oracle_optimality_rate"] == 1.0
    assert res_random["oracle_optimality_rate"] < res_oracle["oracle_optimality_rate"]
    assert res_random["oracle_positions"] == len(states)
