"""Direct tests for harness.coverage — optimality measured as state-space coverage (§C.41). Validated on
tictactoe, whose reachable space is small enough to enumerate EXACTLY (symmetry-reduced), so an optimal agent
must score 1.0 and a weaker one strictly less."""
from __future__ import annotations

import random

import pytest

from games.tictactoe import TicTacToe
from harness.coverage import (blind_spot_concentration, calibrate_reference, coverage_failures, coverage_vs_reference,
                              decided_frontier, failable_keys, optimal_actions, optimal_play_keys, per_state_act,
                              reachable_states, state_coverage)

G = TicTacToe()


def _optimal_agent(s):
    return sorted(optimal_actions(G, s))[0]      # always play a solver-optimal move


def _random_agent_factory(seed):
    return lambda s: random.Random(f"{seed}:{G.state_key(s)}").choice(G.legal_actions(s))


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


def _rotating_agent():
    calls = [0]

    def act(s):
        calls[0] += 1
        legal = G.legal_actions(s)
        return legal[calls[0] % len(legal)]
    return act


class _CountingAgent:
    """Plays better the more it has been asked, like a search agent that keeps its tree across states."""

    def __init__(self):
        self.calls = 0

    def act(self, game, state, rng):
        self.calls += 1
        legal = game.legal_actions(state)
        return sorted(optimal_actions(game, state))[0] if self.calls > 1 else legal[-1]


def test_a_stateful_act_fn_is_REFUSED_because_coverage_would_depend_on_evaluation_order():
    import pytest
    with pytest.raises(ValueError, match="order"):
        state_coverage(G, _rotating_agent(), exact=True)


def test_a_shared_agent_scores_differently_forward_and_reversed_but_per_state_act_does_not():
    states, _ = reachable_states(G, exact=True)
    shared = _CountingAgent()
    forward = [shared.act(G, s, None) for s in states]
    assert forward[0] != sorted(optimal_actions(G, states[0]))[0] or len(G.legal_actions(states[0])) == 1
    act = per_state_act(G, _CountingAgent)
    assert [act(s) for s in states] == list(reversed([act(s) for s in reversed(states)]))
    assert state_coverage(G, act, states=states)["coverage"] < 1.0


def test_per_state_act_hands_every_state_a_fresh_rng_seeded_the_same_way():
    seen = []

    class _RngAgent:
        def act(self, game, state, rng):
            seen.append(rng.random())
            return game.legal_actions(state)[0]
    act = per_state_act(G, _RngAgent, seed=7)
    s = G.initial_state()
    act(s), act(s)
    assert seen[0] == seen[1] == random.Random(7).random()


def test_coverage_failures_names_the_state_the_move_and_the_value_thrown_away():
    from games.tictactoe import TTTState
    s = TTTState(board=(1, 1, 0, 0, 2, 2, 0, 0, 0), to_move=0, winner=None, done=False)
    quiet = TTTState(board=(1, 0, 0, 0, 2, 0, 0, 0, 0), to_move=0, winner=None, done=False)

    def blunders_at_s(state):
        return 6 if state == s else _optimal_agent(state)
    fails = coverage_failures(G, blunders_at_s, [quiet, s])
    assert len(fails) == 1
    f = fails[0]
    assert f["key"] == G.canonical_key(s) and f["move"] == 6 and f["optimal"] == [2, 3]
    assert f["best"] == 1 and f["played"] == -1 and f["lost"] == 2 and f["severity"] == "win->loss"
    assert f["ply"] == 4


def test_coverage_failures_is_empty_for_the_optimal_agent_and_matches_the_coverage_count():
    states, _ = reachable_states(G, exact=True)
    assert coverage_failures(G, _optimal_agent, states) == []
    rnd = _random_agent_factory(3)
    cov = state_coverage(G, rnd, states=states)
    assert len(coverage_failures(G, rnd, states)) == cov["n_states"] - cov["optimal"]


def test_failable_keys_excludes_states_where_every_legal_move_is_optimal():
    states, _ = reachable_states(G, exact=True)
    keys = failable_keys(G, states)
    assert G.canonical_key(G.initial_state()) not in keys
    assert all(len(G.legal_actions(s)) > 1 for s in states if G.canonical_key(s) in keys)
    failed = {f["key"] for f in coverage_failures(G, _random_agent_factory(5), states)}
    assert failed and failed <= keys and len(keys) < len(states)


def test_optimal_play_keys_are_exactly_the_states_reachable_through_optimal_moves_only():
    from games.tictactoe import TTTState
    keys = optimal_play_keys(G)
    assert G.canonical_key(G.initial_state()) in keys
    center_corner = TTTState(board=(2, 0, 0, 0, 1, 0, 0, 0, 0), to_move=0, winner=None, done=False)
    corner_edge = TTTState(board=(1, 2, 0, 0, 0, 0, 0, 0, 0), to_move=0, winner=None, done=False)
    assert G.canonical_key(center_corner) in keys
    assert G.canonical_key(corner_edge) not in keys
    states, _ = reachable_states(G, exact=True)
    for st in states:
        if G.canonical_key(st) in keys:
            for a in optimal_actions(G, st):
                child = G.step(st, a)
                assert G.is_terminal(child) or G.canonical_key(child) in keys


def test_identical_failures_across_seeds_are_CONCENTRATED_far_beyond_the_null():
    universe = set(range(200))
    same = {3, 17, 42, 99}
    r = blind_spot_concentration([set(same), set(same), set(same) | {150}, set(same), {3, 17, 42}], universe)
    assert r["p"] < 0.001 and r["ratio"] > 10
    assert set(r["recurring"]) == same
    assert r["n_seeds"] == 5 and r["counts"][3] == 5


def test_failures_spread_across_different_states_are_NOT_concentrated():
    universe = set(range(200))
    spread = [set(range(i * 10, i * 10 + k)) for i, k in enumerate((4, 7, 3, 6, 5))]
    r = blind_spot_concentration(spread, universe)
    assert r["observed_pairs"] == 0 and r["p"] > 0.5 and r["recurring"] == []


def test_concentration_is_judged_against_the_FAILABLE_universe_not_the_whole_space():
    import pytest
    with pytest.raises(ValueError, match="universe"):
        blind_spot_concentration([{1, 2}, {1, 500}], set(range(100)))
    small = blind_spot_concentration([{1, 2}, {1, 3}, {2, 3}], {1, 2, 3, 4})
    big = blind_spot_concentration([{1, 2}, {1, 3}, {2, 3}], set(range(1000)))
    assert small["p"] > 0.2 and big["p"] < 0.01


def test_concentration_needs_at_least_two_training_seeds():
    import pytest
    with pytest.raises(ValueError, match="seeds"):
        blind_spot_concentration([{1, 2}], set(range(10)))


def test_decided_frontier_does_NOT_credit_a_move_that_throws_the_proven_result_away():
    states, _ = reachable_states(G, exact=True)
    df = decided_frontier(G, _random_agent_factory(2), states)
    assert df["decided_frac"] == decided_frontier(G, _optimal_agent, states)["decided_frac"]
    assert 0.0 < df["kept_when_decided"] < 0.9


def _toy_encode(game, state):
    return repr(state.board).encode()


def test_training_visits_maps_every_symmetric_image_back_to_its_canonical_state():
    from games.tictactoe import TTTState
    from harness.coverage import training_visits
    corner = TTTState(board=(1, 0, 0, 0, 0, 0, 0, 0, 0), to_move=1, winner=None, done=False)
    other_corner = TTTState(board=(0, 0, 0, 0, 0, 0, 0, 0, 1), to_move=1, winner=None, done=False)
    center = TTTState(board=(0, 0, 0, 0, 1, 0, 0, 0, 0), to_move=1, winner=None, done=False)
    buffer = [(_toy_encode(G, corner), None, 0.0), (_toy_encode(G, other_corner), None, 0.0),
              (_toy_encode(G, center), None, 0.0), (b"not a position", None, 0.0)]
    r = training_visits(G, buffer, _toy_encode)
    assert r["visits"][G.canonical_key(corner)] == 2
    assert r["visits"][G.canonical_key(center)] == 1
    assert r["unmatched"] == 1


def test_training_visits_REFUSES_an_encoding_that_merges_distinct_positions():
    import pytest
    from harness.coverage import training_visits
    with pytest.raises(ValueError, match="collid"):
        training_visits(G, [], lambda game, s: b"same")


def _onehot(a):
    return [1.0 if i == a else 0.0 for i in range(9)]


def test_label_dose_counts_the_labels_TRAINED_ON_at_target_states_in_each_image_s_own_frame():
    from games.tictactoe import TTTState
    from harness.coverage import label_dose
    win = TTTState(board=(1, 1, 0, 0, 2, 2, 0, 0, 0), to_move=0, winner=None, done=False)
    mirrored = TTTState(board=(0, 1, 1, 2, 2, 0, 0, 0, 0), to_move=0, winner=None, done=False)
    assert G.canonical_key(win) == G.canonical_key(mirrored)
    assert optimal_actions(G, mirrored) == {0, 5}
    other = TTTState(board=(1, 0, 0, 0, 0, 0, 0, 0, 0), to_move=1, winner=None, done=False)
    examples = [(_toy_encode(G, win), _onehot(2), 0.0),
                (_toy_encode(G, mirrored), _onehot(0), 0.0),
                (_toy_encode(G, mirrored), _onehot(6), 0.0),
                (_toy_encode(G, other), _onehot(4), 0.0),
                (b"no such position", _onehot(4), 0.0)]
    d = label_dose(G, examples, _toy_encode, {G.canonical_key(win)})
    k = G.canonical_key(win)
    assert d["per_key"][k]["n"] == 3 and d["per_key"][k]["argmax_ok"] == 2
    assert d["per_key"][k]["opt_mass"] == pytest.approx(2.0)
    assert d["n_target"] == 3 and d["argmax_ok"] == 2 and d["unmatched"] == 1


def test_label_dose_credits_the_optimal_MASS_of_a_soft_label_even_when_its_argmax_is_wrong():
    from games.tictactoe import TTTState
    from harness.coverage import label_dose
    win = TTTState(board=(1, 1, 0, 0, 2, 2, 0, 0, 0), to_move=0, winner=None, done=False)
    soft = [0.0] * 9
    soft[2], soft[3], soft[7] = 0.2, 0.2, 0.6
    d = label_dose(G, [(_toy_encode(G, win), soft, 0.0)], _toy_encode, {G.canonical_key(win)})
    assert d["argmax_ok"] == 0 and d["opt_mass"] == pytest.approx(0.4)


def test_label_dose_REFUSES_an_encoding_shared_by_two_different_positions():
    import pytest as _pytest
    from harness.coverage import label_dose
    with _pytest.raises(ValueError, match="collid"):
        label_dose(G, [], lambda game, s: b"same", set())
