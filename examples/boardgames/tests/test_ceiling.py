"""Direct tests for harness.ceiling — reading a multi-seed coverage-failure dump into the §C.42 localization
verdicts. Fixtures are synthetic evidence whose answer is known by construction, varied in SHAPE (how many seeds,
how failures fall across plies and the manifold) so a verdict cannot pass by accident of one layout."""
from __future__ import annotations

import hashlib
from functools import lru_cache

import pytest

from harness.ceiling import localize_report


def _fail(key, ply, deep_ok=False, prior=True, value=True, severity="win->draw"):
    return {"key": key, "ply": ply, "severity": severity, "deep_ok": deep_ok,
            "prior_prefers_played": prior, "value_prefers_played": value}


def _evidence(seeds, universe=range(100), on_path=range(0, 20), plies=None):
    universe = list(universe)
    plies = plies or {k: 2 + k % 5 for k in universe}
    return {"universe": universe, "optimal_play_keys": list(on_path),
            "plies": [[k, plies[k]] for k in universe], "seeds": seeds}


def _seed(failures, visits=None, coverage=0.99):
    return {"failures": failures, "coverage": coverage, "visits": [[k, v] for k, v in (visits or {}).items()]}


def test_the_same_states_failed_by_every_seed_read_as_SYSTEMATIC_but_four_off_path_states_are_not_ENRICHMENT():
    shared = [_fail(k, 2 + k % 5) for k in (40, 41, 60, 77)]
    seeds = [_seed(list(shared)) for _ in range(8)]
    r = localize_report(_evidence(seeds))
    assert r["systematic"]["verdict"] is True and r["systematic"]["p"] < 0.001
    assert r["off_manifold"]["failing_off"] == 4 and r["off_manifold"]["verdict"] is False
    assert sorted(x["key"] for x in r["recurring"]) == [40, 41, 60, 77]


def test_off_manifold_needs_MORE_than_the_base_rate_of_off_path_states():
    fails = [_fail(k, 3) for k in range(50, 90)]
    seeds = [_seed(fails[i::3]) for i in range(3)]
    r = localize_report(_evidence(seeds, universe=range(100), on_path=range(0, 50)))
    assert r["off_manifold"]["base_rate"] == 0.5 and r["off_manifold"]["failing_off"] == 40
    assert r["off_manifold"]["verdict"] is True


def test_deep_search_fixing_almost_every_failure_reads_as_a_BUDGET_problem_not_the_net():
    seeds = [_seed([_fail(k, 3, deep_ok=True) for k in range(s * 5, s * 5 + 5)]) for s in range(6)]
    r = localize_report(_evidence(seeds))
    assert r["net_not_budget"]["fix_rate"] == 1.0
    assert r["net_not_budget"]["verdict"] is False and r["net_not_budget"]["budget_fixes"] is True


def test_deep_search_fixing_none_reads_as_the_NET_being_wrong():
    seeds = [_seed([_fail(k, 3) for k in range(s * 5, s * 5 + 5)]) for s in range(6)]
    r = localize_report(_evidence(seeds))
    assert r["net_not_budget"]["fix_rate"] == 0.0 and r["net_not_budget"]["verdict"] is True


def test_starved_compares_visits_only_WITHIN_the_same_seed_and_ply():
    plies = {k: 2 if k < 50 else 6 for k in range(100)}
    visits = {k: (100 if k < 50 else 1) for k in range(100)}
    late = [_fail(k, 6) for k in range(50, 60)]
    seeds = [_seed(late, visits) for _ in range(4)]
    r = localize_report(_evidence(seeds, plies=plies))
    assert r["starved"]["verdict"] is False and r["starved"]["p"] > 0.3
    starved = {**visits, **{k: 0 for k in range(50, 60)}}
    r2 = localize_report(_evidence([_seed(late, starved) for _ in range(4)], plies=plies))
    assert r2["starved"]["verdict"] is True


def test_the_family_alpha_is_split_across_the_four_claims():
    seeds = [_seed([_fail(1, 2)]), _seed([_fail(2, 2)])]
    r = localize_report(_evidence(seeds), alpha=0.05)
    assert r["alpha_each"] == pytest.approx(0.0125)


def test_a_failure_outside_the_universe_is_refused():
    with pytest.raises(ValueError, match="universe"):
        localize_report(_evidence([_seed([_fail(500, 2)]), _seed([_fail(1, 2)])]))


def test_the_decomposition_counts_prior_and_value_errors_separately():
    fails = [_fail(1, 2, prior=True, value=False), _fail(2, 3, prior=False, value=True),
             _fail(3, 4, prior=True, value=True)]
    r = localize_report(_evidence([_seed(fails), _seed(fails[:1])]))
    d = r["decomposition"]
    assert d["pairs"] == 4 and d["prior_prefers_played"] == 3 and d["value_prefers_played"] == 2
    assert d["severity"] == {"win->draw": 4}
    assert d["by_ply"] == {2: 2, 3: 1, 4: 1}


def test_a_low_fix_rate_on_FEW_failures_is_inconclusive_not_a_verdict():
    seeds = [_seed([_fail(1, 3, deep_ok=True), _fail(2, 3)]), _seed([_fail(3, 3), _fail(4, 3)])]
    nb = localize_report(_evidence(seeds))["net_not_budget"]
    assert nb["fix_rate"] == 0.25
    assert nb["verdict"] is False and nb["budget_fixes"] is False


def test_budget_attribution_is_CONFOUNDED_when_search_alone_fixes_the_same_failures():
    fails = [dict(_fail(k, 4, deep_ok=True), control_deep_ok=True) for k in range(12)]
    nb = localize_report(_evidence([_seed(fails[:6]), _seed(fails[6:])]))["net_not_budget"]
    assert nb["budget_fixes"] is True and nb["control_fix_rate"] == 1.0
    assert nb["confounded"] is True and nb["attributable"] is False


def test_budget_attribution_stands_when_search_alone_does_NOT_fix_them():
    fails = [dict(_fail(k, 4, deep_ok=True), control_deep_ok=False) for k in range(12)]
    nb = localize_report(_evidence([_seed(fails[:6]), _seed(fails[6:])]))["net_not_budget"]
    assert nb["confounded"] is False and nb["attributable"] is True


def test_budget_attribution_WITHOUT_a_search_alone_control_is_never_attributable():
    fails = [_fail(k, 4, deep_ok=True) for k in range(12)]
    nb = localize_report(_evidence([_seed(fails[:6]), _seed(fails[6:])]))["net_not_budget"]
    assert nb["confounded"] is None and nb["attributable"] is False


def test_a_search_alone_control_on_FEW_failures_cannot_settle_the_confound():
    fails = [dict(_fail(k, 4, deep_ok=True), control_deep_ok=k < 2) for k in range(3)]
    nb = localize_report(_evidence([_seed(fails[:2]), _seed(fails[2:])]))["net_not_budget"]
    assert nb["control_fix_rate"] == pytest.approx(2 / 3)
    assert nb["confounded"] is None and nb["attributable"] is False


def _arm(seeds, cfg=None, fp="fp1", **per_seed):
    base_cfg = {"iterations": 6, "selfplay": 48, "opening_plies": 0, "opening_zero_frac": 0.0}
    rows = []
    for i, s in enumerate(seeds):
        row = {"seed": s, "coverage": per_seed["coverage"][i], "policy_coverage": per_seed["policy"][i],
               "failures": [{"key": k} for k in per_seed["fails"][i]],
               "policy_fail_keys": list(per_seed.get("pfails", [[]] * len(seeds))[i]),
               "visits": [[k, v] for k, v in per_seed.get("visits", [{}] * len(seeds))[i].items()]}
        rows.append(row)
    return {"training_fingerprint": fp, "config": {**base_cfg, **(cfg or {}), "seeds": list(seeds)},
            "universe": list(range(100)), "seeds": rows}


def _target(seeds=(1, 2, 3), keys=(5, 6, 7, 8)):
    return {"universe": list(range(100)), "seeds": [{"seed": s, "failures": [{"key": k} for k in keys]}
                                                   for s in seeds]}


TREAT = {"opening_plies": 2, "opening_zero_frac": 0.5}


def test_a_treatment_that_fixes_the_target_states_in_every_seed_is_SUPPORTED():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    base = _arm(seeds, coverage=[0.98] * 10, policy=[0.94 + 0.001 * i for i in range(10)],
                fails=[[5, 6, 7, 30 + i] for i in range(10)])
    treat = _arm(seeds, cfg=TREAT, coverage=[0.99] * 10, policy=[0.96 + 0.001 * i for i in range(10)],
                 fails=[[30 + i] for i in range(10)])
    r = ab_report(base, treat, _target())
    assert r["treatment"] == TREAT
    assert r["target"]["verdict"] is True and r["target"]["mean_delta"] == pytest.approx(0.75)
    assert r["policy"]["verdict"] is True and r["coverage"]["verdict"] is True
    assert r["replication"]["verdict"] is True


def test_a_treatment_with_no_consistent_effect_is_NOT_supported():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    wobble = [0.01 if i % 2 else -0.01 for i in range(10)]
    base = _arm(seeds, coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5, 6] for _ in range(10)])
    treat = _arm(seeds, cfg=TREAT, coverage=[0.98 + w for w in wobble], policy=[0.95 + w for w in wobble],
                 fails=[[5, 6] if i % 2 else [5] for i in range(10)])
    r = ab_report(base, treat, _target())
    assert r["coverage"]["verdict"] is False and r["policy"]["verdict"] is False
    assert r["coverage"]["p"] > 0.3


def test_the_ab_REFUSES_a_target_set_chosen_from_the_seeds_it_evaluates():
    from harness.ceiling import ab_report
    seeds = range(1, 11)
    base = _arm(seeds, coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5]] * 10)
    treat = _arm(seeds, cfg=TREAT, coverage=[0.99] * 10, policy=[0.96] * 10, fails=[[]] * 10)
    with pytest.raises(ValueError, match="regression to the mean"):
        ab_report(base, treat, _target(seeds=(1, 2, 3)))


def test_the_ab_REFUSES_arms_that_differ_in_more_than_the_declared_treatment():
    from harness.ceiling import ab_report
    seeds = range(11, 14)
    kw = dict(coverage=[0.98] * 3, policy=[0.95] * 3, fails=[[5]] * 3)
    base = _arm(seeds, **kw)
    with pytest.raises(ValueError, match="differ"):
        ab_report(base, _arm(seeds, cfg={**TREAT, "iterations": 12}, **kw), _target())
    with pytest.raises(ValueError, match="no difference"):
        ab_report(base, _arm(seeds, **kw), _target())


def test_the_ab_REFUSES_unpaired_seeds_and_moved_training_code():
    from harness.ceiling import ab_report
    kw = dict(coverage=[0.98] * 3, policy=[0.95] * 3, fails=[[5]] * 3)
    base = _arm(range(11, 14), **kw)
    with pytest.raises(ValueError, match="seeds"):
        ab_report(base, _arm(range(14, 17), cfg=TREAT, **kw), _target())
    with pytest.raises(ValueError, match="fingerprint"):
        ab_report(base, _arm(range(11, 14), cfg=TREAT, fp="fp2", **kw), _target())


def test_replication_asks_whether_FRESH_baseline_seeds_miss_the_target_beyond_its_share_of_the_universe():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    spread = _arm(seeds, coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[40 + i, 60 + i] for i in range(10)])
    treat = _arm(seeds, cfg=TREAT, coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[40 + i] for i in range(10)])
    rep = ab_report(spread, treat, _target())["replication"]
    assert rep["verdict"] is False and rep["in_target"] == 0 and rep["target_share"] == pytest.approx(0.04)


def test_target_policy_correctness_is_read_from_the_raw_policy_failures():
    from harness.ceiling import ab_report
    seeds = range(11, 14)
    base = _arm(seeds, coverage=[0.98] * 3, policy=[0.95] * 3, fails=[[5]] * 3, pfails=[[5, 6, 7, 8]] * 3)
    treat = _arm(seeds, cfg=TREAT, coverage=[0.98] * 3, policy=[0.95] * 3, fails=[[5]] * 3, pfails=[[5]] * 3)
    r = ab_report(base, treat, _target())
    assert r["target_policy"]["mean_delta"] == pytest.approx(0.75)


def _coverage_ab(deltas):
    from harness.ceiling import ab_report
    seeds = range(11, 11 + len(deltas))
    k = len(deltas)
    base = _arm(seeds, coverage=[0.95] * k, policy=[0.95] * k, fails=[[5]] * k)
    treat = _arm(seeds, cfg=TREAT, coverage=[0.95 + d for d in deltas], policy=[0.95] * k, fails=[[5]] * k)
    return ab_report(base, treat, _target())


def test_a_gain_significant_at_alpha_but_NOT_at_alpha_over_four_is_not_supported():
    cov = _coverage_ab([0.012, -0.004, 0.010, 0.002, 0.009, -0.003, 0.011, 0.001, 0.008, -0.002])["coverage"]
    assert 0.0125 < cov["p"] < 0.05
    assert cov["verdict"] is False


def test_a_consistent_LOSS_is_far_from_significant_as_a_gain():
    cov = _coverage_ab([-0.012, -0.010, -0.011, -0.009, -0.013, -0.010, -0.012, -0.008, -0.011, -0.010])["coverage"]
    assert cov["mean_delta"] < 0 and cov["p"] > 0.99 and cov["verdict"] is False


def test_fresh_seeds_missing_the_target_only_at_its_base_rate_do_NOT_replicate():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    fails = [[20 + (8 * i + j) % 80 for j in range(9)] + ([5 + i] if i < 4 else [30 + (8 * i + 9) % 70])
             for i in range(10)]
    base = _arm(seeds, coverage=[0.9] * 10, policy=[0.9] * 10, fails=fails)
    treat = _arm(seeds, cfg=TREAT, coverage=[0.9] * 10, policy=[0.9] * 10, fails=[[]] * 10)
    rep = ab_report(base, treat, _target())["replication"]
    assert rep["in_target"] > 0 and rep["verdict"] is False


def test_identical_gains_in_every_seed_are_only_as_significant_as_the_number_of_seeds():
    two = _coverage_ab([0.01, 0.01])["coverage"]
    ten = _coverage_ab([0.01] * 10)["coverage"]
    assert two["p"] == pytest.approx(0.25) and two["verdict"] is False
    assert ten["p"] == pytest.approx(1 / 1024) and ten["verdict"] is True


def test_a_single_seed_pair_is_refused_the_unit_is_the_training_run():
    with pytest.raises(ValueError, match="two seed pairs"):
        _coverage_ab([0.05])


def test_manipulation_check_confirms_the_treatment_RAISED_target_state_visits():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    kw = dict(coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5]] * 10)
    base = _arm(seeds, visits=[{5: 0, 6: 0, 7: 0, 8: 0}] * 10, **kw)
    treat = _arm(seeds, cfg=TREAT, visits=[{5: 12, 6: 10, 7: 8, 8: 9}] * 10, **kw)
    m = ab_report(base, treat, _target())["manipulation"]
    assert m["base_mean"] == 0.0 and m["treat_mean"] == pytest.approx(39.0)
    assert m["verdict"] is True and m["p"] == pytest.approx(1 / 1024)


def test_a_treatment_that_did_NOT_reach_the_target_states_fails_the_manipulation_check():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    kw = dict(coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5]] * 10)
    same = [{5: 4, 6: 4, 7: 4, 8: 4}] * 10
    base = _arm(seeds, visits=same, **kw)
    treat = _arm(seeds, cfg=TREAT, visits=same, **kw)
    m = ab_report(base, treat, _target())["manipulation"]
    assert m["verdict"] is False


def test_a_raised_visit_count_with_a_flat_target_is_read_as_starvation_NOT_causal():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    # visits go up in treat, but the same target states stay failed in both arms
    base = _arm(seeds, coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5, 6, 7, 8]] * 10,
                visits=[{5: 0, 6: 0, 7: 0, 8: 0}] * 10)
    treat = _arm(seeds, cfg=TREAT, coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5, 6, 7, 8]] * 10,
                 visits=[{5: 15, 6: 15, 7: 15, 8: 15}] * 10)
    r = ab_report(base, treat, _target())
    assert r["manipulation"]["verdict"] is True and r["target"]["verdict"] is False
    assert r["mechanism_causal"]["mechanism"] == "visits"
    assert r["mechanism_causal"]["manipulation_succeeded"] is True
    assert r["mechanism_causal"]["target_improved"] is False
    assert r["mechanism_causal"]["is_causal"] is False


def test_visits_are_causal_only_when_BOTH_manipulation_and_target_improve():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    # target improves, but visits did NOT rise (a fix that did not come through more exposure)
    same = [{5: 8, 6: 8, 7: 8, 8: 8}] * 10
    base = _arm(seeds, coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5, 6, 7, 8]] * 10, visits=same)
    treat = _arm(seeds, cfg=TREAT, coverage=[0.99] * 10, policy=[0.95] * 10, fails=[[]] * 10, visits=same)
    r = ab_report(base, treat, _target())
    assert r["target"]["verdict"] is True and r["manipulation"]["verdict"] is False
    assert r["mechanism_causal"]["is_causal"] is False


def test_manipulation_counts_only_TARGET_state_visits_not_visits_elsewhere():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    kw = dict(coverage=[0.98] * 10, policy=[0.95] * 10, fails=[[5]] * 10)
    # treat pours visits into NON-target states 90-93; target states 5-8 stay at zero in both arms
    base = _arm(seeds, visits=[{5: 0, 6: 0, 7: 0, 8: 0, 90: 0}] * 10, **kw)
    treat = _arm(seeds, cfg=TREAT, visits=[{5: 0, 6: 0, 7: 0, 8: 0, 90: 99, 91: 99, 92: 99, 93: 99}] * 10, **kw)
    m = ab_report(base, treat, _target())["manipulation"]
    assert m["base_mean"] == 0.0 and m["treat_mean"] == 0.0 and m["verdict"] is False


SIMS = {"train_sims": 200}


def _label_arm(seeds, label_ok, cfg=None):
    kw = dict(coverage=[0.98] * len(seeds), policy=[0.95] * len(seeds), fails=[[5]] * len(seeds))
    arm = _arm(seeds, cfg={"train_sims": 32, **(cfg or {})}, **kw)
    for row, ok in zip(arm["seeds"], label_ok):
        row["target_labels"] = {"label_ok": ok, "prior_anchor": 10 - ok, "search_miss": 0}
    return arm


def test_a_LABEL_manipulation_reads_the_target_label_ok_rate_of_each_arm():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    base = _label_arm(seeds, [2] * 10)
    treat = _label_arm(seeds, [9] * 10, cfg=SIMS)
    m = ab_report(base, treat, _target(), manipulation="labels", treatment_keys=("train_sims",))["manipulation"]
    assert m["kind"] == "labels"
    assert m["base_mean"] == pytest.approx(0.2) and m["treat_mean"] == pytest.approx(0.9)
    assert m["verdict"] is True


def test_a_manipulation_field_MISSING_from_a_seed_is_refused_not_read_as_zero():
    from harness.ceiling import ab_report
    seeds = range(11, 14)
    base = _label_arm(seeds, [2] * 3)
    treat = _label_arm(seeds, [9] * 3, cfg=SIMS)
    del treat["seeds"][1]["target_labels"]
    with pytest.raises(ValueError, match="target_labels"):
        ab_report(base, treat, _target(), manipulation="labels", treatment_keys=("train_sims",))
    del base["seeds"][0]["visits"]
    with pytest.raises(ValueError, match="visits"):
        ab_report(base, _arm(seeds, cfg=TREAT, coverage=[0.98] * 3, policy=[0.95] * 3, fails=[[5]] * 3),
                  _target(), treatment_keys=("train_sims", "opening_plies", "opening_zero_frac"))


def test_an_unknown_manipulation_is_refused():
    from harness.ceiling import ab_report
    seeds = range(11, 14)
    with pytest.raises(ValueError, match="manipulation"):
        ab_report(_label_arm(seeds, [2] * 3), _label_arm(seeds, [9] * 3, cfg=SIMS), _target(),
                  manipulation="vibes", treatment_keys=("train_sims",))


def test_the_causal_readout_names_the_manipulated_mechanism():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    base = _label_arm(seeds, [2] * 10)
    treat = _label_arm(seeds, [9] * 10, cfg=SIMS)
    for row in treat["seeds"]:
        row["failures"] = []
    r = ab_report(base, treat, _target(), manipulation="labels", treatment_keys=("train_sims",))
    c = r["mechanism_causal"]
    assert c["mechanism"] == "labels" and c["manipulation_succeeded"] is True and c["target_improved"] is True
    assert c["is_causal"] is True


def test_the_paired_gain_reports_a_one_sided_LOWER_and_UPPER_confidence_bound_by_inverting_the_sign_flip():
    from harness.ceiling import _paired_gain
    r = _paired_gain([0.10] * 10, 0.05)
    assert r["lower"] == pytest.approx(0.10, abs=1e-6) and r["upper"] == pytest.approx(0.10, abs=1e-6)
    spread = _paired_gain([0.3, -0.1, 0.2, 0.0, 0.1, 0.25, -0.05, 0.15, 0.05, 0.1], 0.05)
    assert spread["lower"] < spread["mean_delta"] < spread["upper"]
    assert spread["lower"] > -0.1 and spread["upper"] < 0.3


def test_the_lower_bound_is_the_edge_of_what_the_test_would_reject():
    from harness.ceiling import _paired_gain
    d = [0.3, -0.1, 0.2, 0.0, 0.1, 0.25, -0.05, 0.15, 0.05, 0.1]
    r = _paired_gain(d, 0.05)
    below = _paired_gain([x - (r["lower"] - 1e-3) for x in d], 0.05)
    above = _paired_gain([x - (r["lower"] + 1e-3) for x in d], 0.05)
    assert below["p"] < 0.05 <= above["p"]


def test_a_null_result_reads_as_an_UPPER_bound_on_the_effect_it_could_have_missed():
    from harness.ceiling import _paired_gain
    r = _paired_gain([0.06, -0.06, 0.12, 0.0, 0.06, -0.06, 0.0, 0.12, 0.06, 0.0], 0.05)
    assert r["verdict"] is False
    assert 0.03 < r["upper"] < 0.12


def test_the_fast_sign_flip_count_equals_brute_force_enumeration_on_varied_shapes():
    import random as _random
    from harness.ceiling import _signflip_p

    def brute(d):
        obs = sum(d) - 1e-12
        n = len(d)
        return sum(1 for s in range(2 ** n)
                   if sum(abs(x) if (s >> i) & 1 else -abs(x) for i, x in enumerate(d)) >= obs) / 2 ** n
    rng = _random.Random(4)
    for n in (2, 3, 7, 10, 11):
        for _ in range(6):
            d = [rng.choice([0.0, 1 / 17, -1 / 17, 2 / 17, rng.uniform(-0.2, 0.3)]) for _ in range(n)]
            assert _signflip_p(d) == pytest.approx(brute(d))


def test_the_confidence_bounds_do_not_move_with_the_claim_s_own_alpha():
    from harness.ceiling import _paired_gain
    d = [0.3, -0.1, 0.2, 0.0, 0.1, 0.25, -0.05, 0.15, 0.05, 0.1]
    strict, loose = _paired_gain(d, 0.0083), _paired_gain(d, 0.05)
    assert strict["lower"] == loose["lower"] and strict["upper"] == loose["upper"]
    assert _paired_gain(d, 0.05, conf_alpha=0.01)["lower"] < loose["lower"]


def _dose_arm(seeds, argmax_ok, n_target=20, cfg=None, fails=None, mfp="m1"):
    k = len(seeds)
    arm = _arm(seeds, cfg={"train_sims": 32, **(cfg or {})}, coverage=[0.98] * k, policy=[0.95] * k,
               fails=fails or [[5]] * k)
    arm["measurement_fingerprint"] = mfp
    for row, ok in zip(arm["seeds"], argmax_ok):
        row["written_labels"] = [{"pass": p, "examples": 8000, "n_target": n_target, "argmax_ok": ok,
                                  "opt_mass": float(ok), "per_key": {}} for p in (1, 2, 3)]
    return arm


def test_an_explicit_alpha_each_is_the_bar_every_verdict_is_judged_at():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    base = _dose_arm(seeds, [2] * 10)
    treat = _dose_arm(seeds, [9] * 10, cfg=SIMS)
    r = ab_report(base, treat, _target(), manipulation="dose", treatment_keys=("train_sims",), alpha_each=0.0005)
    assert r["alpha_each"] == 0.0005
    assert r["manipulation"]["p"] == pytest.approx(1 / 1024) and r["manipulation"]["verdict"] is False


def test_the_DOSE_manipulation_counts_correct_labels_trained_on_summed_over_passes():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    m = ab_report(_dose_arm(seeds, [2] * 10), _dose_arm(seeds, [9] * 10, cfg=SIMS), _target(),
                  manipulation="dose", treatment_keys=("train_sims",))["manipulation"]
    assert m["kind"] == "dose" and m["base_mean"] == 6 and m["treat_mean"] == 27 and m["verdict"] is True


def test_the_migration_guard_flags_failures_that_MOVE_outside_the_target():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    base = _dose_arm(seeds, [2] * 10, fails=[[5, 6, 7]] * 10)
    moved = _dose_arm(seeds, [9] * 10, cfg=SIMS, fails=[[40, 41, 42]] * 10)
    ot = ab_report(base, moved, _target(), manipulation="dose", treatment_keys=("train_sims",))["outside_target"]
    assert ot["base_mean"] == 0 and ot["treat_mean"] == 3
    assert ot["upper"] > 0.5 and ot["holds"] is False
    stayed = _dose_arm(seeds, [9] * 10, cfg=SIMS, fails=[[5]] * 10)
    ot2 = ab_report(base, stayed, _target(), manipulation="dose", treatment_keys=("train_sims",))["outside_target"]
    assert ot2["holds"] is True


def test_arms_measured_by_DIFFERENT_measurement_code_are_refused():
    from harness.ceiling import ab_report
    seeds = range(11, 14)
    with pytest.raises(ValueError, match="measurement"):
        ab_report(_dose_arm(seeds, [2] * 3), _dose_arm(seeds, [9] * 3, cfg=SIMS, mfp="m2"), _target(),
                  manipulation="dose", treatment_keys=("train_sims",))


def test_the_migration_guard_judges_the_UPPER_BOUND_not_a_small_mean():
    from harness.ceiling import ab_report
    seeds = range(11, 21)
    base = _dose_arm(seeds, [2] * 10, fails=[[5]] * 10)
    few = _dose_arm(seeds, [9] * 10, cfg=SIMS, fails=[[5]] * 8 + [[40, 41]] * 2)
    ot = ab_report(base, few, _target(), manipulation="dose", treatment_keys=("train_sims",))["outside_target"]
    assert ot["treat_mean"] - ot["base_mean"] == pytest.approx(0.4)
    assert ot["upper"] > 0.5 and ot["holds"] is False


MIXED = {"opening_plies": 2, "opening_zero_frac": 0.5}
C45_SEEDS = range(21, 31)


def _c45_arm(cfg, fails, dose=5, pfails=None, n_target=20):
    arm = _dose_arm(C45_SEEDS, [dose] * 10, cfg=cfg, fails=fails, n_target=n_target)
    for row, pf in zip(arm["seeds"], pfails or [[5, 6, 7]] * 10):
        row["policy_fail_keys"] = list(pf)
    return arm


def _c45_arms(r200_fails=None, r200_dose=18, r200_pfails=None, rs_fails=None, deep_fails=None, base_fails=None,
              r200_n=20, r32_dose=5):
    stuck = [[5, 6, 7]] * 10
    r = {"reanalyze_frac": 1.0}
    return {"base": _c45_arm({}, base_fails or stuck),
            "mixed": _c45_arm(MIXED, stuck),
            "mixed_deep": _c45_arm({**MIXED, "train_sims": 200}, deep_fails or [[]] * 10, dose=18),
            "mixed_R32": _c45_arm({**MIXED, **r, "reanalyze_sims": 32}, stuck, dose=r32_dose),
            "mixed_R200": _c45_arm({**MIXED, **r, "reanalyze_sims": 200}, r200_fails if r200_fails is not None
                                   else [[]] * 10, dose=r200_dose, pfails=r200_pfails or [[]] * 10, n_target=r200_n),
            "mixed_RS": _c45_arm({**MIXED, **r, "reanalyze_sims": 64}, rs_fails or [[]] * 10, dose=18)}


def test_c45_a_label_fix_that_is_delivered_and_does_not_move_is_SUPPORTED_down_the_whole_chain():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(), _target())
    assert r["G1"]["passed"] is True
    assert {k: r["claims"][k]["verdict"] for k in ("A1", "A2", "B1")} == dict.fromkeys(("A1", "A2", "B1"), "supported")
    assert r["claims"]["A1"]["alpha"] == 0.04 and r["claims"]["B1"]["alpha"] == 0.01


def test_c45_the_fixed_sequence_STOPS_at_the_first_claim_that_is_not_supported():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(r200_fails=[[5, 6, 7]] * 10, r200_pfails=[[5, 6, 7]] * 10), _target())
    assert r["claims"]["A1"]["verdict"] == "refuted"
    assert r["claims"]["A2"]["verdict"] == "not_reached"
    assert r["claims"]["B1"]["verdict"] == "supported"


def test_c45_a_fix_without_DELIVERED_labels_is_not_delivered_never_refuted_or_supported():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(r200_dose=5), _target())
    assert r["claims"]["A1"]["verdict"] == "not_delivered"
    assert r["claims"]["A2"]["verdict"] == "not_reached"


def test_c45_delivery_fails_when_any_seed_trained_on_NO_target_label():
    from harness.ceiling import c45_report
    arms = _c45_arms()
    for p in arms["mixed_R200"]["seeds"][3]["written_labels"]:
        p["n_target"], p["argmax_ok"] = 0, 0
    assert c45_report(arms, _target())["claims"]["A1"]["verdict"] == "not_delivered"


def test_c45_delivery_needs_the_R200_labels_to_be_mostly_RIGHT_not_just_more():
    from harness.ceiling import c45_report
    assert c45_report(_c45_arms(r200_dose=18, r200_n=40), _target())["claims"]["A1"]["verdict"] == "not_delivered"


def test_c45_a_fix_that_MOVES_failures_outside_the_target_reads_moved():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(r200_fails=[[40, 41, 42]] * 10), _target())
    assert r["claims"]["A1"]["verdict"] == "moved"
    assert r["claims"]["A2"]["verdict"] == "not_reached"


def test_c45_a_noisy_null_is_INCONCLUSIVE_when_its_upper_bound_does_not_exclude_the_smallest_effect():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(r200_fails=[[]] * 4 + [[5, 6, 7, 8]] * 6), _target())
    a1 = r["claims"]["A1"]
    assert a1["verdict"] == "inconclusive" and a1["upper"] >= a1["sesoi"] and a1["p"] >= 0.04


def test_c45_every_claim_is_STOPPED_when_the_blind_spot_does_not_replicate():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(base_fails=[[40 + i] for i in range(10)]), _target())
    assert r["G1"]["passed"] is False
    assert {c["verdict"] for c in r["claims"].values()} == {"stopped"}


def test_c45_a_claim_whose_arm_was_not_run_reads_not_run():
    from harness.ceiling import c45_report
    arms = _c45_arms()
    del arms["mixed_deep"]
    r = c45_report(arms, _target())
    assert r["claims"]["B1"]["verdict"] == "not_run" and r["claims"]["A1"]["verdict"] == "supported"


def test_c45_delivery_needs_the_correct_label_count_to_RISE_over_the_control():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(r32_dose=18), _target())
    a1 = r["claims"]["A1"]
    assert a1["delivery"]["written_share"] >= 0.85 and a1["delivery"]["every_seed_delivered"] is True
    assert a1["verdict"] == "not_delivered"


def test_a_bound_that_lands_EXACTLY_on_the_migration_margin_holds():
    from harness.ceiling import ab_report
    seeds = range(21, 41)
    base = _dose_arm(seeds, [2] * 20, fails=[[5]] * 20)
    treat = _dose_arm(seeds, [9] * 20, cfg=SIMS, fails=[[5, 40]] * 6 + [[5]] * 14)
    ot = ab_report(base, treat, _target(), manipulation="dose", treatment_keys=("train_sims",))["outside_target"]
    assert ot["upper"] == pytest.approx(0.5, abs=1e-9)
    assert ot["holds"] is True


def test_a_null_whose_upper_bound_EQUALS_the_smallest_effect_is_inconclusive_not_refuted():
    from harness.ceiling import _bound_below
    assert _bound_below(0.5 + 5e-14, 0.5) is False
    assert _bound_below(0.5 - 5e-14, 0.5) is False
    assert _bound_below(0.5 - 1e-3, 0.5) is True
    assert _bound_below(0.5 + 1e-3, 0.5) is False


def test_a_bound_at_or_under_a_margin_is_within_it_despite_float_slack():
    from harness.ceiling import _bound_within
    assert _bound_within(0.5 + 5e-14, 0.5) is True
    assert _bound_within(0.5 + 1e-3, 0.5) is False


def test_a_report_whose_target_differs_from_the_one_the_arms_RECORDED_is_refused():
    from harness.ceiling import ab_report
    seeds = range(11, 14)
    base = _dose_arm(seeds, [2] * 3)
    treat = _dose_arm(seeds, [9] * 3, cfg=SIMS)
    for arm in (base, treat):
        arm["config"]["target_keys"] = [5, 6, 7, 8]
    ab_report(base, treat, _target(), manipulation="dose", treatment_keys=("train_sims",))
    with pytest.raises(ValueError, match="target"):
        ab_report(base, treat, _target(keys=(5, 6)), manipulation="dose", treatment_keys=("train_sims",))


def test_the_c45_family_is_exactly_A1_A2_and_B1_once_A3_was_dropped():
    from harness.ceiling import c45_report
    r = c45_report(_c45_arms(), _target())
    assert set(r["claims"]) == {"A1", "A2", "B1"}


def test_the_descriptive_manipulation_is_judged_at_the_GATE_alpha_so_it_cannot_contradict_G2():
    from harness.ceiling import c45_report
    arms = _c45_arms()
    for row, d in zip(arms["mixed_deep"]["seeds"], [6] * 3 + [5] * 7):
        for p in row["written_labels"]:
            p["argmax_ok"] = d
    r = c45_report(arms, _target())
    b1 = r["claims"]["B1"]
    desc = r["descriptive"]["B1: mixed_deep vs mixed"]
    assert b1["delivery"]["dose_rises"] == desc["manipulation"]["verdict"]
    assert desc["manipulation"]["alpha"] == 0.05


def test_c45_reports_the_delivered_label_SHARE_alongside_the_count():
    from harness.ceiling import c45_report
    d = c45_report(_c45_arms(), _target())["claims"]["A1"]["delivery"]
    assert d["share"]["treat_mean"] == pytest.approx(0.9) and d["share"]["base_mean"] == pytest.approx(0.25)
    assert d["share"]["mean_delta"] == pytest.approx(0.65) and d["share"]["p"] == pytest.approx(1 / 1024)


def test_ab_report_and_c46_share_ONE_pairing_refusal_that_returns_the_paired_rows():
    from harness.ceiling import _pair_arms
    seeds = range(11, 14)
    kw = dict(coverage=[0.98] * 3, policy=[0.95] * 3, fails=[[5]] * 3)
    keys = ("opening_plies", "opening_zero_frac")
    pair = _pair_arms(_arm(seeds, **kw), _arm(seeds, cfg=TREAT, **kw), _target(), keys)
    assert pair["diff"] == ["opening_plies", "opening_zero_frac"] and pair["order"] == [11, 12, 13]
    assert pair["target_keys"] == {5, 6, 7, 8} and sorted(pair["base"]) == sorted(pair["treat"]) == [11, 12, 13]
    with pytest.raises(ValueError, match="beyond the declared"):
        _pair_arms(_arm(seeds, **kw), _arm(seeds, cfg=TREAT, **kw), _target(), ("opening_plies",))


def test_c45_and_c46_judge_a_claim_on_one_ladder():
    from harness.ceiling import _judge
    hit, null, wide = {"p": 0.001, "upper": 2.0}, {"p": 0.5, "upper": 0.2}, {"p": 0.5, "upper": 3.0}
    assert _judge(hit, 0.01, 1.0, False, True) == "not_delivered"
    assert _judge(hit, 0.01, 1.0, True, True) == "supported"
    assert _judge(hit, 0.01, 1.0, True, False) == "moved"
    assert _judge(null, 0.01, 1.0, True, True) == "refuted"
    assert _judge(wide, 0.01, 1.0, True, True) == "inconclusive"
    assert _judge({"p": 0.01, "upper": 0.2}, 0.01, 1.0, True, True) == "refuted"


# §C.46 — synthetic arms in the full evidence schema, on the REAL tic-tac-toe position graph so the sibling closure
# is checked against the game's rules. Shapes vary by seed and pass: self-play key sets differ per (seed, pass), even
# seeds record the dose over every canonical key (zero rows included) while odd seeds record only the keys trained
# on, failure counts vary per seed with an arm-specific stride, and the hold-out arm withholds keys by the hash rule.
C46_SEEDS = tuple(range(41, 51))
LEGACY = {"channels": 32}
RESIDUAL = {"channels": 32, "blocks": 3, "head_hidden": 32, "residual": True}
HOLDOUT = {"mod": 2, "salt": "c46"}
C46_ARM_SPECS = {  # name: (arch, reanalyze_sims, siblings, oracle labels, hold-out)
    "leg_R32": (LEGACY, None, False, False, None),
    "leg_R200": (LEGACY, 200, False, False, None),
    "R32": (RESIDUAL, None, False, False, None),
    "R200": (RESIDUAL, 200, False, False, None),
    "R32S": (RESIDUAL, None, True, False, None),
    "R200S": (RESIDUAL, 200, True, False, None),
    "Rx": (RESIDUAL, None, False, True, None),
    "RxS": (RESIDUAL, None, True, True, None),
    "RxS_H": (RESIDUAL, None, True, True, HOLDOUT),
}
C46_PROFILE = {  # name: (target raw-policy failures, outside raw-policy failures, target eval failures, outside eval)
    "leg_R32": (3, 10, 3, 1), "leg_R200": (1, 8, 1, 1), "R32": (3, 6, 3, 1), "R200": (1, 5, 1, 1),
    "R32S": (1, 3, 1, 0), "R200S": (0, 2, 0, 0), "Rx": (0, 3, 0, 0), "RxS": (0, 1, 0, 0), "RxS_H": (0, 2, 0, 0),
}


@lru_cache(maxsize=1)
def _ttt():
    """The tic-tac-toe position graph built the brute-force way — the children of EVERY raw image of a canonical
    key, not of one representative — so the report's closure is checked against a different computation."""
    from games.tictactoe import TicTacToe
    from harness.coverage import failable_keys, reachable_states
    game = TicTacToe()
    raw, _ = reachable_states(game, exact=True, symmetry=False)
    kids: dict = {}
    for s in raw:
        out = kids.setdefault(game.canonical_key(s), set())
        for a in game.legal_actions(s):
            child = game.step(s, a)
            if not game.is_terminal(child):
                out.add(game.canonical_key(child))
    canon, _ = reachable_states(game, exact=True, symmetry=True)
    by_ply: dict = {}
    for s in canon:
        by_ply.setdefault(game.ply(s), []).append(game.canonical_key(s))
    return game, kids, sorted(failable_keys(game, canon)), by_ply


def _held(k, holdout=HOLDOUT):
    return int(hashlib.sha256(f"{holdout['salt']}:{k!r}".encode()).hexdigest(), 16) % holdout["mod"] == 0


def _ring(sp, holdout=None):
    kids = _ttt()[1]
    out = set().union(*(kids[k] for k in sp)) - set(sp)
    return {k for k in out if holdout is None or not _held(k, holdout)}


def _sp(i, p):
    by = _ttt()[3]
    return sorted({0, *by[1][: 1 + (i + p) % 3], *by[2][(3 * i + p) % 12:: 7], *by[3][(i + 2 * p) % 19:: 11]})


def _t17():
    universe = set(_ttt()[2])
    return [k for k in _ttt()[3][6] if k in universe][5:9]


def _outside():
    t = set(_t17())
    return [k for k in _ttt()[2] if k not in t]


def _c46_arm(name, seeds=C46_SEEDS):
    _game, kids, universe, _by = _ttt()
    arch, sims, sib, oracle, holdout = C46_ARM_SPECS[name]
    t_raw, o_raw, t_eval, o_eval = C46_PROFILE[name]
    tk, out = _t17(), _outside()
    stride = list(C46_ARM_SPECS).index(name) + 1
    residual = arch is RESIDUAL
    cfg = {"game": "tictactoe", "iterations": 6, "selfplay": 48, "train_sims": 32, "eval_sims": 48, "arch": arch,
           "params_expected": 57453 if residual else 12746, "reanalyze_frac": 1.0, "reanalyze_sims": sims,
           "reanalyze_siblings": sib, "sibling_key": "canonical" if sib else None, "sibling_holdout": holdout,
           "steps_matched": sib, "policy_target": "exact_uniform_optimal" if oracle else None, "epochs": 6,
           "batch_size": 64, "lr": 1e-3, "buffer_cap": 8000, "target_keys": list(tk), "seeds": list(seeds)}
    if oracle:
        cfg["oracle_fingerprint"] = "ofp"
    rows = []
    for i, seed in enumerate(seeds):
        pfk = tk[:t_raw] + out[(7 * i) % 50:(7 * i) % 50 + o_raw + (i * stride) % 3]
        fails = tk[:t_eval] + out[(3 * i) % 40:(3 * i) % 40 + o_eval]
        passes, history, ring, sp = [], [], [], []
        for p in range(1, 7):
            sp = _sp(i, p)
            ring = sorted(_ring(sp, holdout)) if sib and p >= 2 else []
            dose = [[k, 8, 8, 0, 0] for k in sp] + [[k, 0, 0, 8, 8] for k in ring]
            if i % 2 == 0:
                dose += [[k, 0, 0, 0, 0] for k in sorted(set(kids) - set(sp) - set(ring))]
            passes.append({"pass": p, "weights_sha": f"{'res' if residual else 'leg'}-{seed}" if p == 1
                           else f"{name}-{seed}-{p}",
                           "policy_fail_keys": pfk + out[len(out) - (6 - p):] if p < 6 else list(pfk), "dose": dose})
            history.append({"iteration": p, "siblings": len(ring), "train_examples": 8 * (len(sp) + len(ring)),
                            "epoch_examples": 8 * len(sp), "steps": 540 + 6 * i + (12 * (i % 4) if sib else 0)})
        low = name in ("leg_R32", "R32")
        written = [{"pass": p, "examples": 8000, "n_target": 20 + p % 2,
                    "argmax_ok": 5 + i % 2 if low else 18 + p % 2, "opt_mass": 0.0, "per_key": []} for p in range(1, 7)]
        rows.append({"seed": seed, "coverage": 1 - len(fails) / 627, "policy_coverage": 1 - len(pfk) / 627,
                     "failures": [{"key": k} for k in fails], "policy_fail_keys": pfk, "written_labels": written,
                     "visits": [[k, 8 * (1 + j % 3)] for j, k in enumerate(sp)],
                     "sibling_visits": [[k, 8] for k in ring], "history": history, "passes": passes,
                     "params": 57453 if residual else 12746})
    return {"training_fingerprint": "tfp", "measurement_fingerprint": "mfp", "started": "2026-09-24T00:00:00+00:00",
            "config": cfg, "universe": list(universe), "optimal_play_keys": [], "plies": [], "seeds": rows}


def _c46_arms(names=tuple(C46_ARM_SPECS), seeds=C46_SEEDS):
    return {n: _c46_arm(n, seeds) for n in names}


def _g0(verdict="passed", fp="tfp"):
    return {"training_fingerprint": fp, "verdict": {"verdict": verdict},
            "summary": [{"arch": "legacy", "epochs": 62, "failures": [5, 8, 1, 6, 3]}]}


def _c46(arms, g0=None, **kw):
    from harness.ceiling import c46_report
    return c46_report(arms, _target(keys=tuple(_t17())), _ttt()[0], _g0() if g0 is None else g0, **kw)


def _verdicts(r):
    return {k: c["verdict"] for k, c in r["claims"].items()}


E_ARMS = ("R200", "R200S", "R32", "R32S")


def test_c46_every_claim_is_SUPPORTED_when_each_treatment_works_and_every_gate_holds():
    r = _c46(_c46_arms())
    assert _verdicts(r) == dict.fromkeys(("E1", "E2", "A1", "L1", "L2", "R1", "R2"), "supported")
    assert set(r) == {"claims", "descriptive"}
    assert {k: (c["chain"], c["alpha"]) for k, c in r["claims"].items()} == {
        "E1": ("E", 0.03), "E2": ("E", 0.03), "A1": ("A", 0.01), "L1": ("R", 0.01), "L2": ("R", 0.01),
        "R1": ("R", 0.01), "R2": ("R", 0.01)}
    ge = r["claims"]["E1"]["gates"]["GE"]
    assert ge["passed"] is True and ge["siblings_every_pass"] is True and ge["control_has_none"] is True
    assert ge["exposure"]["rises"] is True and ge["closure"]["holds"] is True and ge["closure"]["mismatches"] == []
    assert ge["step_ratio"]["within"] is True and ge["sibling_label_share"]["share"] == 1.0
    assert r["claims"]["E2"]["gates"]["GE"]["sibling_label_share"]["min"] is None
    assert r["claims"]["A1"]["gates"]["G0"]["passed"] is True
    assert r["claims"]["R1"]["gates"]["G1-leg"]["passed"] is True and r["claims"]["R1"]["gates"]["G1-res"]["passed"]
    assert r["claims"]["L1"]["gates"]["G2"]["passed"] is True and r["claims"]["L1"]["gates"]["G3"]["holds"] is True
    assert "G3" not in r["claims"]["L2"]["gates"]
    assert "L1: leg_R200 vs leg_R32" in r["descriptive"] and "R2: R200 vs R32" in r["descriptive"]


def test_c46_policy_failure_claims_read_the_gain_as_CONTROL_minus_TREATMENT_failures_per_seed():
    arms = _c46_arms(E_ARMS)
    e1 = _c46(arms)["claims"]["E1"]
    base = [len(s["policy_fail_keys"]) for s in arms["R200"]["seeds"]]
    treat = [len(s["policy_fail_keys"]) for s in arms["R200S"]["seeds"]]
    assert e1["control_failures"] == base and e1["treat_failures"] == treat
    assert e1["control"] == "R200" and e1["treat"] == "R200S"
    assert e1["mean_delta"] == pytest.approx(sum(base) / 10 - sum(treat) / 10)
    assert e1["sesoi"] == pytest.approx(0.25 * sum(base) / 10)
    assert e1["gain_failable_coverage"] == pytest.approx(e1["mean_delta"] / len(_ttt()[2]))
    assert e1["p"] == pytest.approx(1 / 1024)


def _set_failures(arm, counts):
    out = _outside()
    for row, c in zip(arm["seeds"], counts):
        row["policy_fail_keys"] = out[:c]


def test_c46_a_null_whose_upper_bound_EXCLUDES_the_smallest_effect_is_refuted_worded_as_a_bound_and_halts_E():
    arms = _c46_arms(E_ARMS)
    base = [len(s["policy_fail_keys"]) for s in arms["R200"]["seeds"]]
    _set_failures(arms["R200S"], [b - d for b, d in zip(base, [1, -1, 0, 1, -1, 0, 0, 1, -1, 0])])
    r = _c46(arms)
    e1 = r["claims"]["E1"]
    assert e1["verdict"] == "refuted" and e1["upper"] < e1["sesoi"]
    assert e1["statement"] == f"effect bounded below {round(e1['upper'], 3) + 0.0:.3f} failures per seed"
    assert r["claims"]["E2"]["verdict"] == "not_reached"


def test_c46_a_noisy_null_is_INCONCLUSIVE_and_carries_no_bound_statement():
    arms = _c46_arms(E_ARMS)
    base = [len(s["policy_fail_keys"]) for s in arms["R200"]["seeds"]]
    _set_failures(arms["R200S"], [b - d for b, d in zip(base, [6, -5, 4, -6, 5, -4, 0, 3, -3, 1])])
    e1 = _c46(arms)["claims"]["E1"]
    assert e1["verdict"] == "inconclusive" and e1["p"] >= 0.03 and e1["upper"] >= e1["sesoi"]
    assert "statement" not in e1


def _dose(arm, i, p):
    return next(x for x in arm["seeds"][i]["passes"] if x["pass"] == p)["dose"]


def _ge(r, claim="E1"):
    return r["claims"][claim]["gates"]["GE"]


def test_c46_GE_i_a_treatment_pass_WITHOUT_sibling_rows_is_not_delivered():
    arms = _c46_arms(E_ARMS)
    by = _ttt()[3]
    late = sorted(set(by[7]) | set(by[8]))
    assert _ring(late) == set()
    row = arms["R200S"]["seeds"][2]
    next(x for x in row["passes"] if x["pass"] == 4)["dose"] = [[k, 8, 8, 0, 0] for k in late]
    r = _c46(arms)
    ge = _ge(r)
    assert ge["siblings_every_pass"] is False and ge["closure"]["holds"] is True and ge["exposure"]["rises"] is True
    assert ge["passed"] is False and r["claims"]["E1"]["verdict"] == "not_delivered"
    assert r["claims"]["E2"]["verdict"] == "not_reached"


def test_c46_GE_i_a_treatment_seed_with_NO_relabelled_pass_is_not_delivered():
    arms = _c46_arms(E_ARMS)
    for row in (arms["R200S"]["seeds"][5], arms["R200"]["seeds"][5]):
        row["passes"] = [p for p in row["passes"] if p["pass"] == 1]
    ge = _ge(_c46(arms))
    assert ge["siblings_every_pass"] is False and ge["passed"] is False


def test_c46_GE_i_a_CONTROL_with_sibling_rows_is_not_delivered():
    arms = _c46_arms(E_ARMS)
    _dose(arms["R200"], 3, 3).append([_ttt()[3][4][0], 0, 0, 8, 8])
    ge = _ge(_c46(arms))
    assert ge["control_has_none"] is False and ge["siblings_every_pass"] is True and ge["passed"] is False


def test_c46_GE_ii_exposure_that_does_NOT_rise_in_the_final_pass_is_not_delivered():
    arms = _c46_arms(E_ARMS)
    for i in range(len(C46_SEEDS)):
        _dose(arms["R200"], i, 6).extend([k, 8, 8, 0, 0] for k in _ttt()[2])
    ge = _ge(_c46(arms))
    assert ge["exposure"]["rises"] is False and ge["exposure"]["p"] >= 0.05
    assert ge["closure"]["holds"] is True and ge["control_has_none"] is True and ge["passed"] is False


def test_c46_GE_ii_exposure_counts_only_FAILABLE_keys_of_the_FINAL_pass():
    arms = _c46_arms(E_ARMS)
    universe = set(_ttt()[2])
    by = _ttt()[3]
    unfailable = [k for p in (5, 6, 7) for k in by[p] if k not in universe]
    for i in range(len(C46_SEEDS)):
        _dose(arms["R200"], i, 6).extend([k, 8, 8, 0, 0] for k in unfailable)
        _dose(arms["R200"], i, 5).extend([k, 8, 8, 0, 0] for k in _ttt()[2])
    ge = _ge(_c46(arms))
    assert ge["exposure"]["rises"] is True and ge["passed"] is True


def test_c46_GE_iii_a_ring_key_with_NO_sibling_row_breaks_the_closure():
    arms = _c46_arms(E_ARMS)
    dose = _dose(arms["R200S"], 3, 5)
    sib = [row for row in dose if row[3] > 0]
    dose.remove(sib[1])
    ge = _ge(_c46(arms))
    assert ge["closure"]["holds"] is False and ge["passed"] is False
    assert ge["closure"]["mismatches"] == [{"seed": C46_SEEDS[3], "pass": 5, "missing": [sib[1][0]], "extra": []}]
    assert ge["siblings_every_pass"] is True and ge["exposure"]["rises"] is True


def test_c46_GE_iii_a_sibling_row_OUTSIDE_the_ring_breaks_the_closure():
    arms = _c46_arms(E_ARMS)
    sp = _sp(4, 2)
    stray = next(k for k in _ttt()[3][6] if k not in _ring(sp))
    _dose(arms["R200S"], 4, 2).append([stray, 0, 0, 8, 8])
    ge = _ge(_c46(arms))
    assert ge["closure"]["mismatches"] == [{"seed": C46_SEEDS[4], "pass": 2, "missing": [], "extra": [stray]}]
    assert ge["passed"] is False


def test_c46_GE_iii_the_closure_is_checked_on_relabelled_passes_only():
    arms = _c46_arms(E_ARMS)
    for i in range(len(C46_SEEDS)):
        _dose(arms["R200S"], i, 1).extend([k, 0, 0, 8, 0] for k in _ttt()[3][4][:40])
    r = _c46(arms)
    assert _ge(r)["closure"]["holds"] is True and r["claims"]["E1"]["verdict"] == "supported"


def _set_steps(arm, per_seed):
    for row, steps in zip(arm["seeds"], per_seed):
        for h in row["history"]:
            h["steps"] = steps


def test_c46_GE_iv_the_realised_step_ratio_must_sit_in_its_band_INCLUSIVE_of_both_ends():
    arms = _c46_arms(("R200", "R200S"))
    _set_steps(arms["R200"], [100] * 10)
    _set_steps(arms["R200S"], [75, 133] + [100] * 8)
    ge = _ge(_c46(arms))
    assert ge["step_ratio"]["per_seed"][:2] == pytest.approx([0.75, 1.33])
    assert ge["step_ratio"]["within"] is True and ge["passed"] is True
    for bad in (74, 134):
        _set_steps(arms["R200S"], [bad] + [100] * 9)
        ge = _ge(_c46(arms))
        assert ge["step_ratio"]["within"] is False and ge["passed"] is False


def test_c46_GE_iv_a_control_that_recorded_ZERO_steps_cannot_pass_the_ratio():
    arms = _c46_arms(("R200", "R200S"))
    _set_steps(arms["R200"], [0] + [100] * 9)
    ge = _ge(_c46(arms))
    assert ge["step_ratio"]["per_seed"][0] == float("inf") and ge["step_ratio"]["within"] is False


def _sibling_ok(arm, ok, failable):
    universe = set(_ttt()[2])
    for row in arm["seeds"]:
        for p in row["passes"]:
            for d in p["dose"]:
                if d[3] > 0 and (d[0] in universe) == failable:
                    d[4] = ok


def test_c46_GE_v_E1_needs_its_sibling_labels_on_FAILABLE_rows_to_be_mostly_right():
    arms = _c46_arms(E_ARMS)
    _sibling_ok(arms["R200S"], 7, True)
    r = _c46(arms)
    assert _ge(r)["sibling_label_share"]["share"] == pytest.approx(7 / 8) and _ge(r)["passed"] is False
    assert r["claims"]["E1"]["verdict"] == "not_delivered"


def test_c46_GE_v_a_sibling_label_share_of_EXACTLY_the_minimum_passes():
    arms = _c46_arms(("R200", "R200S"))
    universe = set(_ttt()[2])
    for row in arms["R200S"]["seeds"]:
        for p in row["passes"]:
            for d in p["dose"]:
                if d[3] > 0 and d[0] in universe:
                    d[3], d[4] = 10, 9
    ge = _ge(_c46(arms))
    assert ge["sibling_label_share"]["share"] == 0.9 and ge["sibling_label_share"]["holds"] is True


def test_c46_GE_v_reads_failable_rows_on_relabelled_passes_and_binds_E1_ONLY():
    arms = _c46_arms(E_ARMS)
    _sibling_ok(arms["R200S"], 0, False)
    for i in range(len(C46_SEEDS)):
        _dose(arms["R200S"], i, 1).extend([k, 0, 0, 8, 0] for k in _ttt()[2][:40])
    _sibling_ok(arms["R32S"], 0, True)
    r = _c46(arms)
    assert _ge(r)["sibling_label_share"]["share"] == 1.0 and _verdicts(r)["E1"] == "supported"
    assert _ge(r, "E2")["sibling_label_share"]["share"] == 0.0 and _verdicts(r)["E2"] == "supported"


def test_c46_GE_v_is_not_passed_by_an_arm_with_NO_failable_sibling_rows():
    arms = _c46_arms(("R200", "R200S"))
    universe = set(_ttt()[2])
    for row in arms["R200S"]["seeds"]:
        for p in row["passes"]:
            for d in p["dose"]:
                if d[3] > 0 and d[0] in universe:
                    d[3] = d[4] = 0
    ge = _ge(_c46(arms))
    assert ge["sibling_label_share"]["share"] is None and ge["sibling_label_share"]["holds"] is False


def test_c46_a_blind_spot_that_does_not_replicate_on_LEGACY_stops_the_whole_R_chain():
    arms = _c46_arms()
    out = _outside()
    for i, row in enumerate(arms["leg_R32"]["seeds"]):
        row["failures"] = [{"key": k} for k in out[i * 3:i * 3 + 3]]
    r = _c46(arms)
    assert r["claims"]["L1"]["gates"]["G1-leg"]["passed"] is False
    assert [r["claims"][c]["verdict"] for c in ("L1", "L2", "R1", "R2")] == ["stopped"] * 4
    assert _verdicts(r)["E1"] == "supported" and _verdicts(r)["A1"] == "supported"


def test_c46_a_blind_spot_that_does_not_replicate_on_RESIDUAL_stops_R1_and_R2_only():
    arms = _c46_arms()
    out = _outside()
    for i, row in enumerate(arms["R32"]["seeds"]):
        row["failures"] = [{"key": k} for k in out[i * 3:i * 3 + 3]]
    r = _c46(arms)
    assert [r["claims"][c]["verdict"] for c in ("L1", "L2", "R1", "R2")] == ["supported", "supported", "stopped",
                                                                            "stopped"]


def test_c46_A1_is_STOPPED_when_the_offline_capacity_gate_failed_and_no_other_chain_is():
    r = _c46(_c46_arms(), g0=_g0("failed"))
    assert _verdicts(r) == {**dict.fromkeys(("E1", "E2", "L1", "L2", "R1", "R2"), "supported"), "A1": "stopped"}
    assert r["descriptive"]["D7"]["g0"]["passed"] is False


def test_c46_A1_is_NOT_RUN_when_the_offline_capacity_gate_was_not_run():
    r = _c46(_c46_arms(), g0=_g0("not_run"))
    assert r["claims"]["A1"]["verdict"] == "not_run"


def test_c46_a_G0_trained_by_DIFFERENT_code_than_the_arms_is_refused():
    with pytest.raises(ValueError, match="fingerprint"):
        _c46(_c46_arms(("R200",)), g0=_g0(fp="other"))


def test_c46_a_missing_arm_reads_not_run_and_halts_its_chain():
    arms = _c46_arms()
    del arms["leg_R200"]
    r = _c46(arms)
    assert _verdicts(r) == {"E1": "supported", "E2": "supported", "A1": "not_run", "L1": "not_run",
                            "L2": "not_reached", "R1": "not_reached", "R2": "not_reached"}


def test_c46_a_G1_gate_whose_arm_was_not_run_is_unknown_not_failed():
    arms = _c46_arms(("R32", "R200"))
    r = _c46(arms)
    assert r["claims"]["R1"]["verdict"] == "not_reached" and r["claims"]["L1"]["verdict"] == "not_run"
    assert r["claims"]["L1"]["gates"]["G1-leg"] == {"arm": "leg_R32", "passed": None}


def test_c46_R_claims_without_DELIVERED_labels_are_not_delivered():
    arms = _c46_arms()
    for row in arms["leg_R200"]["seeds"]:
        for w in row["written_labels"]:
            w["argmax_ok"] = 5
    r = _c46(arms)
    assert r["claims"]["L1"]["verdict"] == "not_delivered" and r["claims"]["L1"]["gates"]["G2"]["passed"] is False
    assert [r["claims"][c]["verdict"] for c in ("L2", "R1", "R2")] == ["not_reached"] * 3


def test_c46_a_target_fix_that_MOVES_failures_outside_the_target_reads_moved_and_halts_R():
    arms = _c46_arms()
    out = _outside()
    for row in arms["leg_R200"]["seeds"]:
        row["failures"] += [{"key": k} for k in out[-3:]]
    r = _c46(arms)
    assert r["claims"]["L1"]["verdict"] == "moved" and r["claims"]["L1"]["gates"]["G3"]["holds"] is False
    assert [r["claims"][c]["verdict"] for c in ("L2", "R1", "R2")] == ["not_reached"] * 3


def test_c46_a_null_on_a_target_claim_is_refuted_as_a_bound_on_target_correctness():
    arms = _c46_arms()
    for c, t in zip(arms["R32"]["seeds"], arms["R200"]["seeds"]):
        t["failures"] = [dict(f) for f in c["failures"]]
    r = _c46(arms)
    r1 = r["claims"]["R1"]
    assert r1["verdict"] == "refuted" and r1["sesoi"] == pytest.approx(0.5 * 0.75)
    assert r1["statement"] == f"effect bounded below {round(r1['upper'], 3) + 0.0:.3f} target correctness per seed"
    assert r["claims"]["R2"]["verdict"] == "not_reached"


@pytest.mark.parametrize("where, field", [
    *[("arm", f) for f in ("training_fingerprint", "measurement_fingerprint", "started", "config", "universe",
                           "optimal_play_keys", "plies", "seeds")],
    *[("config", f) for f in ("arch", "params_expected", "train_sims", "reanalyze_frac", "reanalyze_sims",
                              "reanalyze_siblings", "sibling_key", "sibling_holdout", "steps_matched",
                              "policy_target", "target_keys", "seeds")],
    *[("seed", f) for f in ("seed", "coverage", "policy_coverage", "failures", "policy_fail_keys", "written_labels",
                            "visits", "sibling_visits", "history", "passes", "params")],
    *[("history", f) for f in ("siblings", "train_examples", "epoch_examples", "steps")],
    *[("pass", f) for f in ("pass", "weights_sha", "policy_fail_keys", "dose")],
])
def test_c46_a_MISSING_field_is_refused_never_read_as_zero(where, field):
    arms = _c46_arms(("R200", "R200S"))
    arm = arms["R200S"]
    record = {"arm": arm, "config": arm["config"], "seed": arm["seeds"][3], "history": arm["seeds"][3]["history"][2],
              "pass": arm["seeds"][3]["passes"][4]}[where]
    del record[field]
    with pytest.raises(ValueError, match=f"no `{field}`"):
        _c46(arms)


def test_c46_a_net_whose_parameter_count_is_not_the_declared_arch_is_refused():
    arms = _c46_arms(("R200", "R200S"))
    arms["R200S"]["seeds"][6]["params"] = 12746
    with pytest.raises(ValueError, match="12746 parameters"):
        _c46(arms)


def test_c46_arms_declared_to_share_pass_1_must_share_its_weights_byte_for_byte():
    arms = _c46_arms(("R32", "R200"))
    arms["R200"]["seeds"][2]["passes"][0]["weights_sha"] = "drifted"
    with pytest.raises(ValueError, match="pass-1 weights"):
        _c46(arms)


def test_c46_the_legacy_to_residual_pair_is_NOT_held_to_a_shared_pass_1():
    arms = _c46_arms(("leg_R200", "R200"))
    first = [arms[a]["seeds"][0]["passes"][0]["weights_sha"] for a in ("leg_R200", "R200")]
    assert first[0] != first[1]
    assert _c46(arms)["claims"]["A1"]["verdict"] == "supported"


def test_c46_a_seed_with_no_pass_1_or_TWO_of_them_is_refused():
    arms = _c46_arms(("R200", "R200S"))
    arms["R200S"]["seeds"][0]["passes"] = arms["R200S"]["seeds"][0]["passes"][1:]
    with pytest.raises(ValueError, match="pass 1"):
        _c46(arms)
    arms = _c46_arms(("R200", "R200S"))
    passes = arms["R200S"]["seeds"][7]["passes"]
    passes.append(dict(passes[0], weights_sha="res-48-rerun"))
    with pytest.raises(ValueError, match="pass 1"):
        _c46(arms)


def test_c46_arms_that_differ_BEYOND_their_declared_treatment_are_refused():
    arms = _c46_arms(("R200", "R200S"))
    arms["R200S"]["config"]["epochs"] = 12
    with pytest.raises(ValueError, match="beyond the declared"):
        _c46(arms)


def test_c46_arms_trained_by_different_code_are_refused():
    arms = _c46_arms(("RxS", "RxS_H"))
    arms["RxS_H"]["training_fingerprint"] = "tfp2"
    with pytest.raises(ValueError, match="training fingerprint"):
        _c46(arms)


def test_c46_an_UNKNOWN_arm_is_refused_rather_than_silently_ignored():
    arms = _c46_arms(("R200",))
    arms["R200s"] = arms["R200"]
    with pytest.raises(ValueError, match="unknown arm"):
        _c46(arms)


@pytest.mark.parametrize("g0", [{}, {"passed": True}, {"training_fingerprint": "tfp", "verdict": "passed"},
                                {"verdict": {"verdict": "passed"}}])
def test_c46_the_capacity_gate_must_be_passed_as_its_EVIDENCE(g0):
    with pytest.raises(ValueError, match="G0"):
        _c46(_c46_arms(("R200",)), g0=g0)


def test_c46_a_dose_key_that_is_not_a_position_of_the_game_is_refused():
    arms = _c46_arms(("R200", "R200S"))
    _dose(arms["R200S"], 1, 3).append([10 ** 9, 8, 8, 0, 0])
    with pytest.raises(ValueError, match="not a reachable"):
        _c46(arms)


def test_the_sibling_ring_of_the_EMPTY_board_is_its_three_first_moves_minus_the_held_out_ones():
    from harness.ceiling import one_ply_children, sibling_ring
    children = one_ply_children(_ttt()[0])
    assert len(children) == 627
    assert sibling_ring(children, [0]) == {3, 7, 163}
    assert _held(7) and not _held(3) and not _held(163)
    assert sibling_ring(children, [0], HOLDOUT) == {3, 163}
    assert sibling_ring(children, [0, 3]) == {7, 163} | (_ttt()[1][3] - {0, 3})


def test_the_sibling_ring_from_ONE_representative_equals_the_ring_over_EVERY_symmetric_image():
    from harness.ceiling import one_ply_children, sibling_ring
    children = one_ply_children(_ttt()[0])
    assert children == _ttt()[1]
    by = _ttt()[3]
    for i in range(6):
        for p in range(1, 7):
            sp = _sp(i, p) + by[5][i:: 17]
            assert sibling_ring(children, sp) == _ring(sp)
            assert sibling_ring(children, sp, HOLDOUT) == _ring(sp, HOLDOUT)
            assert all(not _held(k) for k in sibling_ring(children, sp, HOLDOUT))
    other = {"mod": 3, "salt": "x"}
    for i in range(4):
        sp = _sp(i, 2) + by[4][i:: 13]
        assert sibling_ring(children, sp, other) == _ring(sp, other) != _ring(sp, HOLDOUT)
    late = set(by[7]) | set(by[8])
    assert sibling_ring(children, late) == set()


def test_the_sibling_ring_refuses_a_key_the_game_cannot_reach():
    from harness.ceiling import one_ply_children, sibling_ring
    with pytest.raises(ValueError, match="not a reachable"):
        sibling_ring(one_ply_children(_ttt()[0]), [0, -5])


def test_a_sibling_on_a_HELD_OUT_key_breaks_the_closure_of_a_holdout_arm():
    from harness.ceiling import _closure_mismatches, one_ply_children
    children = one_ply_children(_ttt()[0])
    row = _c46_arm("RxS_H")["seeds"][2]
    assert _closure_mismatches(row, children, HOLDOUT) == []
    held = next(k for k in sorted(_ring(_sp(2, 3))) if _held(k))
    next(p for p in row["passes"] if p["pass"] == 3)["dose"].append([held, 0, 0, 8, 8])
    assert _closure_mismatches(row, children, HOLDOUT) == [{"pass": 3, "missing": [], "extra": [held]}]
    assert _closure_mismatches(_c46_arm("RxS")["seeds"][2], children, HOLDOUT) != []


def test_clopper_pearson_matches_the_exact_binomial_interval():
    from harness.ceiling import clopper_pearson
    for (k, n), (lo, hi) in {(0, 20): (0.0, 0.1684335), (20, 20): (0.8315665, 1.0), (5, 10): (0.1870860, 0.8129140),
                             (15, 20): (0.5089541, 0.9134285), (1, 20): (0.0012651, 0.2487328),
                             (7, 13): (0.2513455, 0.8077676)}.items():
        got = clopper_pearson(k, n)
        assert got[0] == pytest.approx(lo, abs=1e-7) and got[1] == pytest.approx(hi, abs=1e-7)
    assert clopper_pearson(0, 20)[1] == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-12)
    assert clopper_pearson(5, 10, conf=0.90)[0] > clopper_pearson(5, 10)[0]


@pytest.mark.parametrize("k, n", [(-1, 20), (21, 20), (0, 0)])
def test_clopper_pearson_refuses_an_impossible_count(k, n):
    from harness.ceiling import clopper_pearson
    with pytest.raises(ValueError, match="not a binomial count"):
        clopper_pearson(k, n)


def _perfect(arm, k):
    for i, row in enumerate(arm["seeds"]):
        row["policy_fail_keys"] = [] if i < k else _outside()[:1 + i % 2]


@pytest.mark.parametrize("perfect, reading", [(20, "reaches"), (15, "reaches"), (14, "partial"), (6, "partial"),
                                              (5, "does not"), (0, "does not")])
def test_c46_D1_reads_the_exact_label_ring_ceiling_at_15_and_5_of_20_seeds(perfect, reading):
    arms = _c46_arms(("RxS",), seeds=tuple(range(41, 61)))
    _perfect(arms["RxS"], perfect)
    d1 = _c46(arms)["descriptive"]["D1"]
    assert d1 == {"arm": "RxS", "perfect": perfect, "n": 20, "reading": reading}


def test_c46_D1_without_its_arm_reports_it_missing():
    assert _c46(_c46_arms(("R200",)))["descriptive"]["D1"] == {"missing": ["RxS"]}


@pytest.mark.parametrize("arm, perfect, met", [("R200S", 15, True), ("R200S", 14, False), ("R32", 16, True),
                                               ("RxS", 20, False), ("leg_R200", 20, False)])
def test_c46_the_floor_test_is_met_only_by_a_GENERIC_RESIDUAL_arm_at_15_of_20(arm, perfect, met):
    arms = _c46_arms((arm,), seeds=tuple(range(41, 61)))
    _perfect(arms[arm], perfect)
    m = _c46(arms)["descriptive"]["M"]
    assert m["floor_test_met"] is met
    assert m["arms"][arm]["raw_policy_perfect"] == perfect and m["arms"][arm]["n"] == 20


def test_c46_the_milestone_carries_clopper_pearson_intervals_and_the_search_alone_control():
    from harness.ceiling import clopper_pearson
    arms = _c46_arms(("R200S",), seeds=tuple(range(41, 61)))
    _perfect(arms["R200S"], 15)
    for i, row in enumerate(arms["R200S"]["seeds"]):
        row["failures"] = [] if i < 4 else [{"key": _outside()[0]}]
    arms["R200S"]["search_alone"] = {"coverage_eval": 0.93, "policy_coverage": 0.54}
    row = _c46(arms)["descriptive"]["M"]["arms"]["R200S"]
    assert row["raw_policy_ci"] == pytest.approx(list(clopper_pearson(15, 20)))
    assert row["eval_perfect"] == 4 and row["eval_ci"] == pytest.approx(list(clopper_pearson(4, 20)))
    assert row["search_alone"] == {"coverage_eval": 0.93, "policy_coverage": 0.54}
    assert row["generic_residual"] is True
    assert _c46(_c46_arms(("R32",)))["descriptive"]["M"]["arms"]["R32"]["search_alone"] is None


def _fails_of(arm):
    return [len(s["policy_fail_keys"]) for s in arm["seeds"]]


def test_c46_D2_D4_and_D5_are_paired_differences_with_bounds_but_no_verdict():
    arms = _c46_arms()
    d = _c46(arms)["descriptive"]
    f = {n: _fails_of(a) for n, a in arms.items()}
    assert d["D2"]["deltas"] == [x - y for x, y in zip(f["Rx"], f["RxS"])]
    assert d["D4"]["deltas"] == [x - y for x, y in zip(f["R200S"], f["RxS"])]
    assert d["D5"]["deltas"] == [(a - b) - (c - e) for a, b, c, e in zip(f["R200"], f["R200S"], f["R32"], f["R32S"])]
    for item in ("D2", "D4", "D5"):
        assert d[item]["lower"] <= d[item]["mean_delta"] <= d[item]["upper"]
        assert "verdict" not in d[item] and "p" not in d[item]
    partial = _c46(_c46_arms(("R200", "R200S", "R32")))["descriptive"]
    assert partial["D5"] == {"missing": ["R32S"]} and partial["D2"] == {"missing": ["Rx", "RxS"]}


def test_c46_D3_reads_failures_INSIDE_the_hold_out_and_the_generalisation_ratio():
    arms = _c46_arms(("Rx", "RxS", "RxS_H"))
    universe = set(_ttt()[2])
    late = [k for k in _ttt()[3][7] if k in universe]
    h = [k for k in late if _held(k)]
    u = next(k for k in late if not _held(k))
    for row in arms["Rx"]["seeds"]:
        row["policy_fail_keys"] = [h[0], h[1], h[2], h[3], u]
        next(p for p in row["passes"] if p["pass"] == 4)["dose"].append([h[3], 8, 8, 0, 0])
    for row in arms["RxS_H"]["seeds"]:
        row["policy_fail_keys"] = [h[0], h[1], u]
        next(p for p in row["passes"] if p["pass"] == 2)["dose"].append([h[1], 8, 8, 0, 0])
    for row in arms["RxS"]["seeds"]:
        row["policy_fail_keys"] = [h[0], h[4]]
        next(p for p in row["passes"] if p["pass"] == 3)["dose"].append([h[4], 0, 0, 8, 8])
    d3 = _c46(arms)["descriptive"]["D3"]
    assert d3["holdout"] == HOLDOUT
    assert d3["H"]["mean_failures"] == {"Rx": 4, "RxS_H": 2, "RxS": 2}
    assert d3["H"]["generalisation_ratio"] == pytest.approx(1.0)
    ntr = d3["H_never_trained"]
    assert ntr["mean_failures"] == {"Rx": 3, "RxS_H": 1, "RxS": 2}
    assert ntr["generalisation_ratio"] == pytest.approx(2.0)
    assert ntr["key_set"] == "H keys the RxS_H seed never trained on, the same set for every arm"
    assert ntr["mean_keys"] > 0
    for row in arms["RxS"]["seeds"]:
        row["policy_fail_keys"] = [h[0], h[1], h[2], h[3]]
    assert _c46(arms)["descriptive"]["D3"]["H"]["generalisation_ratio"] is None


def test_c46_D6_sums_each_label_probe_and_reports_an_unrecorded_one_as_None_not_zero():
    arms = _c46_arms(("R200", "R32"))
    for i, row in enumerate(arms["R200"]["seeds"]):
        row["target_labels"] = {"label_ok": 10 + i, "prior_anchor": 2, "search_miss": 1}
        row["target_labels_exact_leaf"] = {"0.1": {"label_ok": 12, "search_miss": 1},
                                           "1.0": {"label_ok": 13, "prior_anchor": i % 2}}
    d6 = _c46(arms)["descriptive"]["D6"]
    assert d6["R200"]["target_labels"] == {"label_ok": 145, "prior_anchor": 20, "search_miss": 10}
    assert d6["R200"]["target_labels_exact_leaf"] == {"0.1": {"label_ok": 120, "search_miss": 10},
                                                      "1.0": {"label_ok": 130, "prior_anchor": 5}}
    assert d6["R32"] == {"target_labels": None, "target_labels_exact_leaf": None}
    del arms["R200"]["seeds"][4]["target_labels_exact_leaf"]
    assert _c46(arms)["descriptive"]["D6"]["R200"]["target_labels_exact_leaf"] is None


def test_c46_D7_traces_raw_policy_failures_per_pass():
    arms = _c46_arms(("R200", "leg_R32"))
    d7 = _c46(arms)["descriptive"]["D7"]
    assert d7["g0"]["passed"] is True and d7["g0"]["summary"] == _g0()["summary"]
    final = sum(_fails_of(arms["R200"])) / 10
    assert d7["raw_policy_by_pass"]["R200"] == [[p, pytest.approx(final + 6 - p)] for p in range(1, 7)]
    assert set(d7["raw_policy_by_pass"]) == {"leg_R32", "R200"}


def test_c46_D8_splits_target_cells_into_never_delivered_and_delivered_correct_but_failed():
    arms = _c46_arms(("R200",))
    tk = _t17()
    for i, row in enumerate(arms["R200"]["seeds"]):
        row["policy_fail_keys"] = [tk[0], tk[1], tk[2]]
        dose3 = next(p for p in row["passes"] if p["pass"] == 3)["dose"]
        dose3 += [[tk[0], 12, 9, 8, 8], [tk[1], 12, 8, 8, 8]] + ([[tk[3], 0, 0, 8, 8]] if i % 2 else [])
        next(p for p in row["passes"] if p["pass"] == 1)["dose"].append([tk[2], 8, 8, 0, 0])
    d8 = _c46(arms, notes={tk[2]: "depth-2 gap"})["descriptive"]["D8"]
    assert d8["share"] == 0.85 and d8["notes"] == [[tk[2], "depth-2 gap"]]
    assert d8["arms"]["R200"]["T"] == {"cells": 40, "failed": 30, "never_delivered": 15, "never_delivered_failed": 10,
                                       "delivered_correct": 15, "delivered_correct_failed": 10}
    per_key = dict((k, v) for k, v in d8["arms"]["R200"]["per_key"])
    assert per_key[tk[2]]["never_delivered_failed"] == 10 and per_key[tk[0]]["delivered_correct_failed"] == 10
    assert d8["arms"]["R200"]["P"] is None


def test_c46_P_is_the_set_of_keys_the_reference_arm_fails_in_at_least_THREE_seeds():
    from harness.ceiling import recurrent_keys
    out = _outside()
    ref = {"seeds": [{"policy_fail_keys": [out[0], out[1]]}, {"policy_fail_keys": [out[0], out[1], out[1]]},
                     {"policy_fail_keys": [out[0], out[2]]}, {"policy_fail_keys": [out[3]]}]}
    assert recurrent_keys(ref) == {out[0]}
    assert recurrent_keys(ref, min_seeds=2) == {out[0], out[1]}
    arms = _c46_arms(("R200",))
    d = _c46(arms, reference=ref)["descriptive"]
    assert d["P"] == [out[0]] and d["T"] == sorted(_t17())
    assert d["D8"]["arms"]["R200"]["P"]["cells"] == 10
    assert d["D8"]["arms"]["R200"]["P"]["failed"] == sum(out[0] in s["policy_fail_keys"] for s in arms["R200"]["seeds"])


def test_c46_D9_places_each_failure_as_VISITED_RING_or_two_plies_away_with_intervals():
    from harness.measurement import t_critical
    arms = _c46_arms(("R200",))
    tk = _t17()
    far = [k for k in _ttt()[3][7] if k in set(_ttt()[2])]
    for i, row in enumerate(arms["R200"]["seeds"]):
        row["visits"] = [[3, 16], [163, 0]]
        for p in row["passes"]:
            p["dose"] = [[0, 8, 8, 0, 0], [3, 8, 8, 0, 0], [163, 0, 0, 0, 0]]
        row["policy_fail_keys"] = [3, 7, 163, tk[0]] + far[:i % 3]
    d9 = _c46(arms)["descriptive"]["D9"]["R200"]
    assert d9["all"]["visited"] == {"mean": 1.0, "ci": [1.0, 1.0]}
    assert d9["all"]["ring"]["mean"] == 2.0
    assert d9["in_T"] == {"visited": {"mean": 0.0, "ci": [0.0, 0.0]}, "ring": {"mean": 0.0, "ci": [0.0, 0.0]},
                          "far": {"mean": 1.0, "ci": [1.0, 1.0]}}
    extra = [i % 3 for i in range(10)]
    m = sum(extra) / 10
    sd = (sum((x - m) ** 2 for x in extra) / 9) ** 0.5
    half = t_critical(9) * sd / 10 ** 0.5
    assert d9["out_T"]["far"]["mean"] == pytest.approx(m)
    assert d9["out_T"]["far"]["ci"] == pytest.approx([m - half, m + half])
    assert d9["all"]["far"]["mean"] == pytest.approx(1 + m)
    assert "in_P" not in d9


def test_c46_D9_splits_by_P_when_a_reference_is_given():
    arms = _c46_arms(("R200",))
    ref = {"seeds": [{"policy_fail_keys": [3]}] * 3}
    for row in arms["R200"]["seeds"]:
        for p in row["passes"]:
            p["dose"] = [[0, 8, 8, 0, 0], [3, 8, 8, 0, 0]]
        row["policy_fail_keys"] = [3, 7]
    d9 = _c46(arms, reference=ref)["descriptive"]["D9"]["R200"]
    assert d9["in_P"]["visited"]["mean"] == 1.0 and d9["in_P"]["ring"]["mean"] == 0.0
    assert d9["out_P"]["visited"]["mean"] == 0.0 and d9["out_P"]["ring"]["mean"] == 1.0


def test_c46_D9_gives_no_interval_for_a_single_seed():
    d9 = _c46(_c46_arms(("R200",), seeds=(41,)))["descriptive"]["D9"]["R200"]
    assert d9["all"]["far"]["ci"] is None and d9["all"]["far"]["mean"] >= 0


def test_a_seed_MISSING_its_raw_policy_failures_is_refused_not_read_as_a_perfect_prior():
    from harness.ceiling import ab_report
    seeds = range(11, 14)
    kw = dict(coverage=[0.98] * 3, policy=[0.95] * 3, fails=[[5]] * 3)
    base, treat = _arm(seeds, **kw), _arm(seeds, cfg=TREAT, **kw)
    del treat["seeds"][2]["policy_fail_keys"]
    with pytest.raises(ValueError, match="policy_fail_keys"):
        ab_report(base, treat, _target())


def test_c46_GE_fails_when_the_siblings_never_CHANGED_training():
    arms = _c46_arms()
    control = {r["seed"]: r for r in arms["R200"]["seeds"]}
    for row in arms["R200S"]["seeds"]:
        for p in row["passes"]:
            if p["pass"] == 2:
                p["weights_sha"] = next(q for q in control[row["seed"]]["passes"] if q["pass"] == 2)["weights_sha"]
    e1 = _c46(arms)["claims"]["E1"]
    assert e1["verdict"] == "not_delivered"
    assert e1["gates"]["GE"]["siblings_changed_training"] is False


def test_localize_report_reads_an_arm_with_NO_failures_as_nothing_starved_instead_of_crashing():
    seeds = [_seed([], visits={k: 1 for k in range(10)}) for _ in range(3)]
    r = localize_report(_evidence(seeds))
    assert r["starved"]["verdict"] is None and r["starved"]["p"] is None
    assert r["systematic"]["verdict"] is None
