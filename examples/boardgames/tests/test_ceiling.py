"""Direct tests for harness.ceiling — reading a multi-seed coverage-failure dump into the §C.42 localization
verdicts. Fixtures are synthetic evidence whose answer is known by construction, varied in SHAPE (how many seeds,
how failures fall across plies and the manifold) so a verdict cannot pass by accident of one layout."""
from __future__ import annotations

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
