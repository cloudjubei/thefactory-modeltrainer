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
    p1_conversion,
    sample_solvable_positions,
    sim_scaling_curve,
)
from harness.agents import RandomAgent
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


def test_sim_scaling_curve_reports_strength_per_budget():
    # The strength-per-COMPUTE headline: conversion vs a fixed reference at each sim budget. A model that ignores
    # its sim budget (the near-perfect oracle plays identically at every budget) yields a perfectly FLAT curve —
    # all strength is in the policy, not bought with search — and as first player it crushes a random reference.
    from harness.solver import NearPerfectOracle

    game = Connect4()
    curve = sim_scaling_curve(game, lambda sims: NearPerfectOracle(depth=8, solve_endgame=22),
                              lambda: RandomAgent(), sims_list=(2, 8), games=4, seed=0)
    assert [p["sims"] for p in curve["points"]] == [2, 8]
    assert all(0.0 <= p["rate"] <= 1.0 for p in curve["points"])
    assert curve["low_sim_rate"] == curve["points"][0]["rate"]
    assert curve["high_sim_rate"] == curve["points"][-1]["rate"]
    # identical play at both budgets ⇒ flatness is EXACTLY 0 and auc == the (high) shared rate
    assert curve["flatness"] == 0.0 and curve["auc"] == curve["low_sim_rate"] and curve["low_sim_rate"] >= 0.75


def test_p1_conversion_opening_plies_diversifies_the_lines():
    # The ladder-rigor fix: with DETERMINISTIC agents and no random openings, every game is the SAME line, so a
    # rung's rate is one repeated game (the depth-4-vs-depth-6 inversion artifact). `opening_plies` diversifies.
    class _Leftmost:  # fully deterministic — always the leftmost legal column
        def act(self, game, state, rng):
            return game.legal_actions(state)[0]

    game = Connect4()
    lf = lambda: _Leftmost()
    det = p1_conversion(game, lf, lf, games=10, seed=1, opening_plies=0)  # one line ×10 → a single outcome bucket
    assert sum(1 for v in (det["wins"], det["draws"], det["losses"]) if v > 0) == 1
    div = p1_conversion(game, lf, lf, games=10, seed=1, opening_plies=6)  # random openings → diverse lines
    assert sum(1 for v in (div["wins"], div["draws"], div["losses"]) if v > 0) >= 2


def test_verify_forced_win_conversion_is_an_exact_proof_without_the_opening_wall():
    # FAST exact proof: optimal play converts PROVEN forced wins vs the EXACT solver from few-empty roots (each
    # solve is milliseconds, not the from-opening minutes); a random model fails to convert them. This is the
    # runnable exact near-optimality evidence the opening wall otherwise blocks.
    from harness.benchmark import sample_forced_win_roots, verify_forced_win_conversion
    from harness.solver import OracleAgent
    from harness.agents import RandomAgent

    game = Connect4()
    roots = sample_forced_win_roots(game, n=3, empties=12, seed=1)
    assert len(roots) == 3 and all(sum(1 for v in r.board if v == 0) == 12 for r in roots)
    opt = verify_forced_win_conversion(game, lambda: OracleAgent(), n_roots=3, empties=12, seed=1)
    assert opt["rate"] == 1.0 and opt["converted"] == 3  # perfect play converts every proven forced win
    weak = verify_forced_win_conversion(game, lambda: RandomAgent(), n_roots=3, empties=12, seed=1)
    assert weak["rate"] < 1.0  # a random model throws proven wins away → exact suboptimality certificate


def test_lbr_screen_profiles_exploitability_by_refuter_depth():
    # §C.8 #14 LBR cheap-screen: the agent plays BOTH seats vs depth-k restricted best responders; the per-depth
    # loss rate is the always-on exploitability gauge (a near-optimal policy stays hard to exploit as k rises).
    from harness.agents import RandomAgent
    from harness.benchmark import lbr_screen
    from harness.solver import OracleAgent

    game = Connect4()
    r = lbr_screen(game, lambda: RandomAgent(), depths=[1, 2], n_openings=2, opening_plies=1, seed=3)
    assert [row["depth"] for row in r["by_depth"]] == [1, 2]
    for row in r["by_depth"]:
        for seat in ("as_p1", "as_p2"):
            s = row[seat]
            assert abs(s["win"] + s["draw"] + s["loss"] - 1.0) < 1e-9
        assert 0.0 <= row["exploit_rate"] <= 1.0
    assert r["games_per_depth"] == 4  # 2 openings x both seats
    # Determinism: the screen is a fixed-seed instrument, so two runs must agree exactly.
    again = lbr_screen(game, lambda: RandomAgent(), depths=[1, 2], n_openings=2, opening_plies=1, seed=3)
    assert again == r
    # A STRONG agent is unexploitable by a depth-1 refuter from the canonical opening. NearPerfectOracle, not
    # OracleAgent: the exact solver from the EMPTY BOARD is the minutes-long opening wall (it hung the suite).
    from harness.solver import NearPerfectOracle

    strong = lbr_screen(game, lambda: NearPerfectOracle(depth=8, solve_endgame=22),
                        depths=[1], n_openings=1, opening_plies=0, seed=0)
    assert strong["by_depth"][0]["as_p1"]["loss"] == 0.0
