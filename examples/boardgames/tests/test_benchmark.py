"""The distance-to-optimal benchmark (harness/benchmark.py): the oracle scores a perfect rate, a weaker policy
does not — so `oracle_optimality_rate` is a real, monotone 'how close to solved' gauge. Corpus is late-game
(high `min_moves`) so the exact solves stay fast."""
import random

from games.connect4 import Connect4
from harness.benchmark import (
    evaluate_optimality,
    exact_reference,
    optimality_rate,
    optimality_trace,
    sample_solvable_positions,
)
from harness.solver import OracleAgent, optimal_columns


def test_optimality_trace_localises_the_first_blunder_and_counts_verified_plies():
    game = Connect4()
    ref = lambda s: [3]  # injected ground truth: column 3 is the only optimal move everywhere (no slow solve)
    good = optimality_trace(game, lambda s: 3, ref, max_plies=5)
    assert good["first_blunder_ply"] is None and good["verified_plies"] == 5
    bad = optimality_trace(game, lambda s: game.legal_actions(s)[0], ref, max_plies=5)
    assert bad["first_blunder_ply"] == 0  # plays column 0 on the empty board → first deviation at ply 0


def test_exact_reference_is_none_without_ground_truth_and_exact_in_the_endgame():
    game = Connect4()
    # No book + no cheap solve in the opening → None (the trace treats that ply as UNVERIFIED, not a blunder).
    assert exact_reference(game, book=None, max_empty=0)(game.initial_state(random.Random(0))) is None
    # A late position within the cheap-solve threshold → the exact optimal set.
    late = sample_solvable_positions(game, n=1, min_moves=30, seed=3)[0]
    ref = exact_reference(game, book=None, max_empty=42)(late)
    assert ref is not None and set(ref) == set(optimal_columns(late))


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
