"""§C.9 MEASUREMENT INTEGRITY — regression tests for the four errors that produced a phantom result.

Each error below actually happened and cost us a wrong conclusion; each test pins the guard that prevents it:
  E1 SELECTION-ON-THE-TEST-SET: the promotion gate probed the same roots the final scorecard reported.
  E2 WINNER'S CURSE: reporting max-over-40-checkpoints on 32 roots as if unbiased (0.969 vs a true ~0.85).
  E3 UNPAIRED / MIXED-PROVENANCE COMPARISON: net A's selected-best vs net B's final -> a phantom difference.
  E4 UNDERPOWERED CLAIM: ranking 8 archs on 16 roots, where +-1 root = 0.06 and the true spread was ~0.08.
"""
import pytest

from harness.measurement import (
    MEASUREMENT_SEEDS,
    SELECTION_SEEDS,
    assert_seed_roles_disjoint,
    expected_selected_max,
    mcnemar_exact,
    min_roots_for,
    report_rate,
    wilson_interval,
)


def test_selection_and_measurement_seeds_are_disjoint():
    # E1 guard. The gate selects on SELECTION_SEEDS, scorecards report on MEASUREMENT_SEEDS; overlap means the
    # selector is choosing on the exam paper. Mutation guard: this must fail if anyone unions the two pools.
    assert SELECTION_SEEDS and MEASUREMENT_SEEDS
    assert SELECTION_SEEDS.isdisjoint(MEASUREMENT_SEEDS)
    assert_seed_roles_disjoint(selection_seed=1013, measurement_seed=7)
    with pytest.raises(ValueError):
        assert_seed_roles_disjoint(selection_seed=7, measurement_seed=7)  # the exact bug we shipped
    with pytest.raises(ValueError):
        assert_seed_roles_disjoint(selection_seed=99, measurement_seed=99)


def test_wilson_interval_brackets_and_widens_as_n_shrinks():
    lo, hi = wilson_interval(31, 32)
    assert lo < 31 / 32 <= hi and lo > 0.8          # 0.969 on 32 roots is NOT precise
    lo16, hi16 = wilson_interval(13, 16)
    lo128, hi128 = wilson_interval(104, 128)
    assert (hi16 - lo16) > (hi128 - lo128) * 1.8    # n=16 is far wider than n=128
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(5, 0)


def test_expected_selected_max_quantifies_the_winners_curse():
    # E2 guard: best-of-k on n roots inflates the REPORTED max well above the true rate. Our case: true ~0.85,
    # k=40 checkpoints, n=32 roots -> a reported ~0.96 is EXPECTED, not evidence of a great net.
    infl = expected_selected_max(true_rate=0.85, n_roots=32, k=40)
    assert 0.93 < infl < 1.0
    assert infl > 0.85 + 0.07
    # more candidates or fewer roots => more inflation; a single candidate => no inflation
    assert expected_selected_max(0.85, 32, 100) > infl > expected_selected_max(0.85, 32, 2)
    assert expected_selected_max(0.85, 16, 40) > expected_selected_max(0.85, 128, 40)
    assert abs(expected_selected_max(0.85, 256, 1) - 0.85) < 0.02


def test_mcnemar_exact_matches_hand_computed_cases():
    # E3 guard: comparisons must be PAIRED on identical roots.
    a = [1] * 15 + [0] * 5 + [1] * 94 + [0] * 14
    b = [0] * 15 + [1] * 5 + [1] * 94 + [0] * 14
    r = mcnemar_exact(a, b)
    assert r["only_a"] == 15 and r["only_b"] == 5 and r["discordant"] == 20
    assert 0.03 < r["p"] < 0.05                      # our measured 15-vs-5 => p ~= 0.041
    tie = mcnemar_exact([1, 0, 1, 0], [1, 0, 1, 0])
    assert tie["discordant"] == 0 and tie["p"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact([1, 0], [1, 0, 1])             # unequal length = not paired


def test_min_roots_for_powers_the_claim():
    # E4 guard: how many roots to detect a delta. Detecting the ~0.02 arch differences we "ranked" on needs
    # vastly more than the 16 roots we used; 16 roots can only detect a huge gap.
    assert min_roots_for(delta=0.02, base=0.85) > 1000
    assert min_roots_for(delta=0.15, base=0.85) < 200
    assert min_roots_for(delta=0.05, base=0.85) > min_roots_for(delta=0.10, base=0.85)


def test_report_rate_refuses_a_naked_selected_number():
    # Every reported rate carries n, seed, CI and its provenance. A SELECTED number must be labelled and
    # must surface the inflation estimate -- reporting it bare is what produced "0.969 BEST EVER".
    r = report_rate(converted=31, n=32, seed=7, selected_from=40, true_rate_hint=0.85)
    assert r["selected"] is True and r["candidates"] == 40
    assert "expected_selected_max" in r and r["expected_selected_max"] > 0.93
    assert r["ci"][0] < r["rate"] <= r["ci"][1]
    assert "SELECTED" in r["caveat"].upper()
    clean = report_rate(converted=107, n=128, seed=99)
    assert clean["selected"] is False and clean["caveat"] == ""
    with pytest.raises(ValueError):
        report_rate(converted=31, n=32, seed=7, selected_from=0)   # k must be >= 1


def test_promotion_margin_blocks_noise_but_admits_decisive_gains():
    # The exact numbers from the failure: incumbent .833 on a 12-root probe. The +0.084 that our gate chased
    # (promoting a net 0.063 WORSE) must be refused; a decisive +0.167 must pass; and a 512-root probe must
    # resolve gains a 12-root probe cannot.
    from harness.measurement import promotion_margin

    m12 = promotion_margin(0.833, 12)
    assert 0.833 + m12 > 0.917          # the real bad promotion is now blocked
    assert 0.833 + m12 < 1.0            # but the gate is not frozen — a decisive gain still promotes
    m512 = promotion_margin(0.833, 512)
    assert m512 < m12 / 5               # margin shrinks with sqrt(n): bigger probes resolve smaller true gains
    assert 0.833 + m512 < 0.917         # +0.084 IS believable at n=512
    with pytest.raises(ValueError):
        promotion_margin(0.5, 0)


def test_homogeneity_refuses_to_rank_an_underpowered_sweep():
    # §C.10 BUILD #2, live-fire on OUR OWN screen: 8 archs, 16 roots each, conversions 10/8/13/12/11/12/10/10.
    # The sweep must REFUSE to emit a ranking (chi2 p ~ 0.67) instead of crowning 03 — which is what we did.
    from harness.measurement import rank_or_refuse

    screen = {"01": (10, 16), "02": (8, 16), "03": (13, 16), "04": (12, 16),
              "05": (11, 16), "06": (12, 16), "07": (10, 16), "08": (10, 16)}
    r = rank_or_refuse(screen)
    assert r["ranked"] is False
    assert r["homogeneity_p"] > 0.20          # no evidence any arm differs
    assert r["required_n"] > 16               # and here is the n it would actually take
    assert "REFUS" in r["verdict"].upper()

    # A genuinely separated sweep DOES rank.
    clear = {"tiny": (40, 256), "big": (200, 256)}
    r2 = rank_or_refuse(clear)
    assert r2["ranked"] is True and r2["order"][0] == "big"

    # A null arm (a ~1% param twin) sets the noise floor. The case it exists for is a gap that IS statistically
    # significant yet smaller than the spread between two architecturally-equivalent arms — significant != real.
    sig_but_meaningless = {"a": (2000, 4000), "b": (2200, 4000)}   # 0.500 vs 0.550, p << 0.05
    assert rank_or_refuse(sig_but_meaningless)["ranked"] is True    # significant on its own
    r3 = rank_or_refuse(sig_but_meaningless, null_arm_spread=0.08)  # but inside the sweep's own noise floor
    assert r3["ranked"] is False and "null-arm" in r3["verdict"].lower()
