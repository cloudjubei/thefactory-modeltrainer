"""§C.13 GRADED + PAIRED CONVERSION — more bits per root, and pairing that cannot be got wrong.

Motivation (measured 2026-09-04): the binary metric yields ~1 bit per root and only ~51% of roots carry any
signal, so resolving a 0.062 architecture effect needed 7 training seeds at n=128. Roots cost seconds; runs cost
hours. Two fixes, both here:
  G1 the exact defender was DETERMINISTIC (centre-first tie-break), so games_per_root>1 replayed the SAME game —
     zero extra information, and conversion was only ever tested against ONE optimal defence out of many.
  G2 pairing was done by hand (and I got it wrong once), so it becomes structural: one call, identical roots.
"""
import random

import pytest

from games.connect4 import Connect4
from harness.benchmark import (
    graded_conversion,
    paired_conversion,
    sample_forced_win_roots,
)
from harness.solver import OracleAgent, RandomizedOracleAgent


def test_randomized_oracle_still_plays_only_optimal_moves():
    # It must remain PERFECT — it varies only among moves that are EQUALLY optimal, never weaker ones.
    # LATE positions only: an exact solve near the opening is the minutes-long wall, and a unit test must not
    # pay it (the same trap that hung the LBR test).
    from harness.benchmark import sample_solvable_positions
    from harness.solver import move_values

    game = Connect4()
    agent = RandomizedOracleAgent()
    for s in sample_solvable_positions(game, n=3, min_moves=30, seed=2):
        vals = move_values(s, weak=False)
        best = max(vals.values())
        for i in range(8):
            assert vals[agent.act(game, s, random.Random(i))] == best, "randomized oracle played a sub-optimal move"


def test_randomized_oracle_actually_varies_where_the_deterministic_one_cannot():
    # The whole point of G1: on a position with several equally-optimal replies the defence must differ by seed.
    from harness.solver import move_values

    from harness.benchmark import sample_solvable_positions

    game = Connect4()
    multi = None
    for s in sample_solvable_positions(game, n=40, min_moves=28, seed=11):  # late => solves are milliseconds
        vals = move_values(s, weak=False)
        if sum(1 for v in vals.values() if v == max(vals.values())) >= 2:
            multi = s
            break
    assert multi is not None, "no multi-optimal position found to test"
    det = {OracleAgent().act(game, multi, random.Random(i)) for i in range(10)}
    rnd = {RandomizedOracleAgent().act(game, multi, random.Random(i)) for i in range(10)}
    assert len(det) == 1                 # deterministic: identical every time (the old behaviour)
    assert len(rnd) >= 2                 # randomized: samples the optimal-defence space


def test_graded_conversion_returns_fractions_not_just_zero_or_one():
    game = Connect4()
    roots = sample_forced_win_roots(game, 3, empties=12, seed=1)
    r = graded_conversion(game, lambda: OracleAgent(), roots, games_per_root=4, seed=1, max_empty=22)
    assert len(r["scores"]) == 3
    assert all(0.0 <= x <= 1.0 for x in r["scores"])
    assert r["rate"] == 1.0                       # perfect play converts every optimal defence
    assert r["binary_rate"] == 1.0
    from harness.agents import RandomAgent
    weak = graded_conversion(game, lambda: RandomAgent(), roots, games_per_root=4, seed=1, max_empty=22)
    assert weak["rate"] < 1.0
    # graded must be at least as informative: it distinguishes "lost every defence" from "lost one of four"
    assert any(0.0 < s < 1.0 for s in weak["scores"]) or weak["rate"] != weak["binary_rate"]


def test_paired_conversion_scores_every_arm_on_identical_roots():
    # G2: pairing is structural — one call, one root set, so a mixed-root comparison is not expressible.
    from harness.agents import RandomAgent

    game = Connect4()
    r = paired_conversion(game, {"oracle": lambda: OracleAgent(), "random": lambda: RandomAgent()},
                          n_roots=3, empties=12, seed=5, games_per_root=2, max_empty=22)
    assert set(r["arms"]) == {"oracle", "random"}
    assert len(r["arms"]["oracle"]["scores"]) == len(r["arms"]["random"]["scores"]) == 3
    assert r["roots_id"] == "e12_s5_n3_g2"        # identifies the exact root family + games/root
    assert r["arms"]["oracle"]["rate"] > r["arms"]["random"]["rate"]


def test_root_sampler_shuffles_so_a_PREFIX_is_representative():
    # The old sampler returned roots in GENERATION order, and a prefix was measured to be systematically easier
    # (mean conversion .804 on roots 0-31 vs .727 on 32-255) — gates and quick checks all read prefixes.
    game = Connect4()
    full = sample_forced_win_roots(game, 12, empties=12, seed=9)
    unshuffled = sample_forced_win_roots(game, 12, empties=12, seed=9, shuffle=False)
    keys = [game.canonical_key(s) for s in full]
    assert sorted(keys) == sorted(game.canonical_key(s) for s in unshuffled)  # same SET of roots
    assert keys != [game.canonical_key(s) for s in unshuffled]                # different ORDER
    assert keys == [game.canonical_key(s) for s in sample_forced_win_roots(game, 12, empties=12, seed=9)]


def test_opening_wall_guard_fails_fast_instead_of_hanging():
    # GUARD for a mistake I made THREE times. The solver refuses an opening-scale solve during tests (limit set
    # in conftest), so the wall raises immediately and names the fix instead of hanging the suite for hours.
    import random

    from harness import solver
    from harness.solver import move_values

    game = Connect4()
    empty = game.initial_state(random.Random(0))          # 42 empties — the wall
    with pytest.raises(RuntimeError, match="opening wall"):
        move_values(empty, weak=False)
    # ...and the guard is scoped: a LATE position still solves normally.
    from harness.benchmark import sample_solvable_positions
    late = sample_solvable_positions(game, n=1, min_moves=30, seed=1)[0]
    assert move_values(late, weak=False) != {}
    # ...and production is unlimited by default (0 = off), so book building is unaffected.
    assert isinstance(solver.MAX_SOLVE_EMPTIES, int)
