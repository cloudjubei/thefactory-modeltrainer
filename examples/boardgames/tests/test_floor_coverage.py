"""Direct tests for harness/floor_coverage.py — the pre-registered §C.48 T5 verdicts: do solver-free COVERAGE levers
(siblings two moves deep, more random opening moves, and — without augmentation — a sibling key that treats every
orientation as its own position) take the fixed process to a raw-perfect tic-tac-toe net, and are the failures that
remain concentrated where the net was never trained? Synthetic arms vary perfection, coverage and integrity."""
from __future__ import annotations

import pytest

from harness.floor_coverage import (T5_ARMS, T5_ERA, T5_MEASUREMENT_FP, T5_SEEDS, T5_REPRO_SEED, coverage_report,
                                    mechanism)
from harness.floor_settle import T4_ARMS, T4_ITERATIONS, T4_PASSES, T4_POSITIONS, T4_SETTLE_EPOCHS, T4_SETTLE_LR_FINAL


def _history(depth):
    rows = [{"iteration": i + 1, "siblings": 0 if i == 0 else 200 * depth, "merged": 900, "self_agreement": 0.9,
             "sibling_rings": [] if i == 0 else [200] * depth} for i in range(T4_PASSES - 1)]
    return rows + [{"iteration": "settle", "epochs": T4_SETTLE_EPOCHS, "lr_final": T4_SETTLE_LR_FINAL,
                    "self_agreement": 0.9}]


def _row(seed, depth, final=0, trained=3000, failures_untrained=None):
    fu = final if failures_untrained is None else failures_untrained
    curve = [final + (T4_PASSES - i) for i in range(1, T4_PASSES + 1)]
    curve[-1] = final
    return {"seed": seed, "params": 57453, "positions": T4_POSITIONS, "strict_failures_per_pass": curve,
            "final_failing_positions": list(range(final)), "games_per_pass": [48] * T4_ITERATIONS + [0],
            "history": _history(depth),
            "coverage": {"trained": trained, "untrained": T4_POSITIONS - trained,
                         "failures_trained": final - fu, "failures_untrained": fu}}


def _arm(name, perfect=10, **kw):
    depth = T5_ARMS[name].get("sibling_depth", 1)
    rows = [_row(s, depth, final=0 if i < perfect else 1 + i % 4, **kw) for i, s in enumerate(T5_SEEDS)]
    return {"training_fingerprint": T5_ERA, "measurement_fingerprint": T5_MEASUREMENT_FP,
            "config": {**T5_ARMS[name], "seeds": list(T5_SEEDS)}, "seeds": rows}


def _t4_augment():
    return {"seeds": [{"seed": s, "strict_failures_per_pass": [s] * T4_PASSES, "final_failing_keys": [s]}
                      for s in T5_SEEDS]}


def _repro(t4=None):
    row = next(r for r in (t4 or _t4_augment())["seeds"] if r["seed"] == T5_REPRO_SEED)
    return {"training_fingerprint": T5_ERA, "config": {**T4_ARMS["augment"], "seeds": [T5_REPRO_SEED]},
            "seeds": [{"seed": T5_REPRO_SEED, "strict_failures_per_pass": list(row["strict_failures_per_pass"]),
                       "final_failing_keys": list(row["final_failing_keys"])}]}


def _all(perfect=10, **kw):
    return {name: _arm(name, perfect, **kw) for name in T5_ARMS}


def _report(arms=None, repro=None, t4=None):
    t4 = t4 or _t4_augment()
    return coverage_report(arms if arms is not None else _all(), repro if repro is not None else _repro(t4), t4)


def test_the_registered_arms_are_T4s_recipe_plus_exactly_one_coverage_lever_each():
    assert set(T5_ARMS) == {"augment_sib2", "augment_open6", "no_augment_rawkey", "no_augment_sib2",
                            "no_augment_open6"}
    base = {name: "augment" if name.startswith("augment") else "no_augment" for name in T5_ARMS}
    extra = {name: {k: v for k, v in T5_ARMS[name].items() if T4_ARMS[base[name]].get(k) != v} for name in T5_ARMS}
    assert extra == {"augment_sib2": {"sibling_depth": 2}, "augment_open6": {"sibling_depth": 1, "opening_plies": 6},
                     "no_augment_rawkey": {"sibling_depth": 1}, "no_augment_sib2": {"sibling_depth": 2},
                     "no_augment_open6": {"sibling_depth": 1, "opening_plies": 6}}
    assert all(set(T4_ARMS[base[n]]) <= set(T5_ARMS[n]) for n in T5_ARMS)
    assert T5_SEEDS == tuple(range(311, 321)) and T5_REPRO_SEED in T5_SEEDS


@pytest.mark.parametrize("perfect,verdict", [(10, "supported"), (8, "supported"), (7, "inconclusive"),
                                             (6, "inconclusive"), (5, "refuted"), (0, "refuted")])
def test_each_arm_is_supported_at_8_of_10_refuted_at_5_inconclusive_between(perfect, verdict):
    arms = _all()
    arms["no_augment_sib2"] = _arm("no_augment_sib2", perfect)
    r = _report(arms)
    assert r["integrity"] == [] and r["no_augment_sib2"]["verdict"] == verdict
    assert r["no_augment_sib2"]["perfect"] == perfect and r["augment_open6"]["verdict"] == "supported"


def test_the_verdict_reads_the_final_pass_after_the_settle():
    arms = _all()
    row = arms["augment_sib2"]["seeds"][0]
    row["strict_failures_per_pass"] = [0] * (T4_PASSES - 1) + [2]
    row["final_failing_positions"] = [1, 2]
    row["coverage"].update({"failures_untrained": 2})
    assert _report(arms)["augment_sib2"]["perfect"] == 9


def _set(path, value):
    def apply(bundle):
        target = bundle
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("arms", "augment_sib2", "training_fingerprint"), "52ea07e577cb"),
    "measurement": _set(("arms", "no_augment_open6", "measurement_fingerprint"), "0" * 12),
    "depth": _set(("arms", "no_augment_sib2", "config", "sibling_depth"), 1),
    "openings": _set(("arms", "augment_open6", "config", "opening_plies"), 2),
    "seeds": _set(("arms", "no_augment_rawkey", "config", "seeds"), list(range(301, 311))),
    "rings before any relabel": _set(("arms", "augment_sib2", "seeds", 0, "history", 0, "sibling_rings"), [5]),
    "a ring missing in the run": _set(("arms", "augment_sib2", "seeds", 3, "history", 5, "sibling_rings"), [400]),
    "coverage does not partition the positions": _set(("arms", "augment_open6", "seeds", 2, "coverage", "trained"), 10),
    "coverage failures disagree": _set(("arms", "no_augment_sib2", "seeds", 9, "coverage", "failures_trained"), 7),
    "coverage missing": lambda b: b["arms"]["no_augment_rawkey"]["seeds"][4].pop("coverage"),
    "a T4 check fails": _set(("arms", "augment_sib2", "seeds", 1, "games_per_pass"), [48] * T4_PASSES),
    "arm missing": lambda b: b["arms"].pop("augment_open6"),
    "repro curve differs": _set(("repro", "seeds", 0, "strict_failures_per_pass"), [0] * T4_PASSES),
    "repro failing keys differ": _set(("repro", "seeds", 0, "final_failing_keys"), [0]),
    "repro era": _set(("repro", "training_fingerprint"), "52ea07e577cb"),
    "repro recipe": _set(("repro", "config", "sibling_depth"), 2),
    "repro seed": _set(("repro", "seeds", 0, "seed"), 312),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_a_broken_integrity_condition_makes_every_verdict_NOT_RUN(name):
    t4 = _t4_augment()
    bundle = {"arms": _all(), "repro": _repro(t4)}
    BREAKAGES[name](bundle)
    r = coverage_report(bundle["arms"], bundle["repro"], t4)
    assert r["integrity"] and all(r[a]["verdict"] == "not_run" for a in T5_ARMS)
    assert r["mechanism"]["verdict"] == "not_run"
    assert not any("could not be read" in p for p in r["integrity"])


@pytest.mark.parametrize("junk", [None, {}, {"augment_sib2": None}])
def test_unreadable_evidence_is_NOT_RUN_never_an_exception(junk):
    r = coverage_report(junk, _repro(), _t4_augment())
    assert all(r[a]["verdict"] == "not_run" for a in T5_ARMS) and r["mechanism"]["verdict"] == "not_run"


def _nets(n, fails_u, fails_t, untrained=500):
    return [{"trained": T4_POSITIONS - untrained, "untrained": untrained, "failures_trained": fails_t,
             "failures_untrained": fails_u} for _ in range(n)]


def test_the_mechanism_is_supported_when_failures_concentrate_on_untrained_positions_net_by_net():
    r = mechanism(_nets(12, 20, 2))
    assert r["verdict"] == "supported" and r["nets"] == 12 and r["untrained_higher"] == 12
    assert r["rate_ratio"] == pytest.approx((240 / 6000) / (24 / 48240)) and r["sign_p"] == 1 / 4096
    assert mechanism(_nets(11, 20, 2) + _nets(1, 0, 5))["sign_p"] == 13 / 4096


def test_a_consistent_but_modest_concentration_is_inconclusive_not_support():
    r = mechanism(_nets(12, 3, 6))
    assert r["untrained_higher"] == 12 and 2 < r["rate_ratio"] < 5 and r["verdict"] == "inconclusive"


def test_the_mechanism_is_refuted_when_trained_positions_fail_as_often():
    assert mechanism(_nets(12, 2, 16))["verdict"] == "refuted"


def test_a_high_pooled_ratio_carried_by_few_nets_is_not_support():
    nets = _nets(1, 400, 0) + _nets(11, 0, 1)
    r = mechanism(nets)
    assert r["rate_ratio"] > 5 and r["untrained_higher"] == 1 and r["verdict"] == "inconclusive"


def test_a_tie_counts_against_the_mechanism_and_perfect_nets_are_not_units():
    r = mechanism(_nets(12, 20, 2)[:11] + _nets(1, 0, 0) + _nets(1, 1, 9, untrained=452))
    assert r["nets"] == 12 and r["untrained_higher"] == 11


def test_too_few_failing_nets_to_judge_the_mechanism_is_NOT_RUN():
    r = mechanism(_nets(9, 20, 0) + _nets(20, 0, 0))
    assert r["verdict"] == "not_run" and r["nets"] == 9


def test_no_failures_on_trained_positions_gives_an_unbounded_ratio_not_a_crash():
    r = mechanism(_nets(12, 3, 0))
    assert r["rate_ratio"] == float("inf") and r["verdict"] == "supported"


def test_the_report_pools_the_mechanism_over_every_arm():
    arms = _all(perfect=0, trained=4000, failures_untrained=None)
    r = _report(arms)
    assert r["mechanism"]["nets"] == 50 and r["mechanism"]["verdict"] == "supported"


def test_descriptives_give_coverage_and_the_share_of_failures_off_the_trained_set():
    arms = _all(perfect=0, trained=4000)
    d = _report(arms)["no_augment_rawkey"]["descriptives"]
    assert d["mean_trained_positions"] == 4000 and d["share_of_failures_untrained"] == 1.0
    assert len(d["final_failures"]) == 10
