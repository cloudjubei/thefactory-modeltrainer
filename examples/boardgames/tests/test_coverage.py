"""Direct tests for harness.coverage — optimality measured as state-space coverage (§C.41). Validated on
tictactoe, whose reachable space is small enough to enumerate EXACTLY (symmetry-reduced), so an optimal agent
must score 1.0 and a weaker one strictly less."""
from __future__ import annotations

import random

from games.tictactoe import TicTacToe
from harness.coverage import (calibrate_reference, coverage_vs_reference, decided_frontier, optimal_actions,
                              reachable_states, state_coverage)

G = TicTacToe()


def _optimal_agent(s):
    return sorted(optimal_actions(G, s))[0]      # always play a solver-optimal move


def _random_agent_factory(seed):
    rng = random.Random(seed)
    return lambda s: rng.choice(G.legal_actions(s))


def test_optimal_actions_picks_the_immediate_win():
    from games.tictactoe import TTTState
    # LEGAL state (X=2, O=2, X to move): the immediate win at 2 AND the fork at 3 (double threat 2 & 6) both
    # force a win, so BOTH are optimal under the weak value; the losing replies (6,7,8) are excluded.
    s = TTTState(board=(1, 1, 0, 0, 2, 2, 0, 0, 0), to_move=0, winner=None, done=False)
    opt = optimal_actions(G, s)
    assert 2 in opt and opt == {2, 3} and opt.isdisjoint({6, 7, 8})


def test_optimal_actions_must_block_the_opponents_win():
    from games.tictactoe import TTTState
    # LEGAL state (X=2, O=1, O to move): X threatens 2 (has 0,1); O must block at 2 or lose -> 2 is optimal
    s = TTTState(board=(1, 1, 0, 0, 2, 0, 0, 0, 0), to_move=1, winner=None, done=False)
    assert optimal_actions(G, s) == {2}


def test_reachable_states_is_exactly_enumerable_and_symmetry_shrinks_it():
    full, complete_full = reachable_states(G, exact=True, symmetry=False)
    canon, complete_canon = reachable_states(G, exact=True, symmetry=True)
    assert complete_full and complete_canon                 # tictactoe fits
    assert len(canon) < len(full)                           # the dihedral group collapses the space
    assert len(full) > 2000 and len(canon) < len(full) / 2  # ~5478 raw states -> a few hundred canonical


def test_an_optimal_agent_covers_the_whole_state_space():
    cov = state_coverage(G, _optimal_agent, exact=True)
    assert cov["complete"] and cov["coverage"] == 1.0        # perfect play = 100% coverage
    assert all(rate == 1.0 for rate in cov["by_ply"].values())


def test_a_random_agent_covers_strictly_less():
    cov = state_coverage(G, _random_agent_factory(0), exact=True)
    assert 0.0 < cov["coverage"] < 1.0                       # a random player is optimal only sometimes
    # and it is WORSE in the midgame than at the very start (where many first moves are all fine)
    assert cov["n_states"] > 100


def test_coverage_can_be_measured_on_a_SAMPLE_for_a_large_space():
    # the sampled mode (for games too big to enumerate) still returns a rate on the states it reached
    cov = state_coverage(G, _optimal_agent, exact=False, sample_playouts=200)
    assert cov["complete"] is False and cov["coverage"] == 1.0 and cov["n_states"] > 50


def test_decided_frontier_credits_proven_non_loss_states():
    states, _ = reachable_states(G, exact=True)
    df = decided_frontier(G, _optimal_agent, states)
    # tictactoe is a draw, so from most states the mover can force a non-loss; an optimal agent keeps it every time
    assert df["decided_frac"] > 0.5 and df["kept_when_decided"] == 1.0
    # but NOT every state is non-lost: reachable positions exist where the mover is already theoretically lost
    # (the opponent has a fork), so they must be EXCLUDED from the decided frontier
    assert df["decided_frac"] < 1.0


def test_calibrate_reference_scores_a_perfect_reference_at_one_and_a_random_one_low():
    # a reference that plays solver-optimal must calibrate at exact_agreement 1.0 (false_optimal 0); a random
    # reference must score well below 1 — this is the trust number for using the reference off-solver.
    states, _ = reachable_states(G, exact=True)
    states = [s for s in states if len(G.legal_actions(s)) > 1][:300]
    good = calibrate_reference(G, _optimal_agent, states=states)
    bad = calibrate_reference(G, _random_agent_factory(0), states=states)
    assert good["exact_agreement"] == 1.0 and good["false_optimal"] == 0.0
    assert bad["exact_agreement"] < 0.95
    assert abs(good["exact_agreement"] + good["false_optimal"] - 1.0) < 1e-9


def test_coverage_vs_reference_matches_the_reference_move():
    states, _ = reachable_states(G, exact=True)
    states = states[:200]
    # an agent that IS the reference covers 1.0 against it; a different agent covers less
    ref = _optimal_agent
    same = coverage_vs_reference(G, ref, ref, states)
    diff = coverage_vs_reference(G, _random_agent_factory(1), ref, states)
    assert same["coverage_vs_reference"] == 1.0
    assert diff["coverage_vs_reference"] < 1.0
