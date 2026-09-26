"""Direct tests for harness/floor_settle.py — the pre-registered §C.48 T4 verdicts: with the process fixes T2 pointed
at (a buffer of unique positions, a settling pass with a decaying learning rate), does the generic self-play process
give a RAW net optimal at every raw tic-tac-toe position? Synthetic arms vary how many seeds end perfect, which
integrity condition breaks, and how the solver-free self-agreement reading lines up with the true failures."""
from __future__ import annotations

import pytest

from harness.floor_settle import (T4_ARMS, T4_ERA, T4_ITERATIONS, T4_MEASUREMENT_FP, T4_PASSES, T4_POSITIONS,
                                  T4_SEEDS, T4_SETTLE_EPOCHS, T4_SETTLE_LR_FINAL, settle_report)


def _history(passes=T4_PASSES, agreement=1.0):
    rows = [{"iteration": i + 1, "siblings": 0 if i == 0 else 200, "merged": 900, "self_agreement": agreement}
            for i in range(passes - 1)]
    return rows + [{"iteration": "settle", "epochs": T4_SETTLE_EPOCHS, "lr_final": T4_SETTLE_LR_FINAL,
                    "self_agreement": agreement}]


def _row(seed, final_failures=0, passes=T4_PASSES, positions=T4_POSITIONS):
    curve = [max(0, final_failures + (passes - i) * 3) for i in range(1, passes + 1)]
    curve[-1] = final_failures
    return {"seed": seed, "params": 57453, "positions": positions, "strict_failures_per_pass": curve,
            "final_failing_positions": list(range(final_failures)), "games_per_pass": [48] * T4_ITERATIONS + [0],
            "history": _history(passes)}


def _arm(name, perfect=10, seeds=T4_SEEDS, **config):
    rows = [_row(s, final_failures=0 if i < perfect else 1 + i % 4) for i, s in enumerate(seeds)]
    return {"training_fingerprint": T4_ERA, "measurement_fingerprint": T4_MEASUREMENT_FP,
            "config": {**T4_ARMS[name], **config, "seeds": list(seeds)}, "seeds": rows}


def _both(perfect_aug=10, perfect_raw=10):
    return {"augment": _arm("augment", perfect_aug), "no_augment": _arm("no_augment", perfect_raw)}


def test_the_registered_recipe_is_R200S_with_the_three_process_fixes_on_fresh_seeds():
    from harness.floor_converge import T2_ARMS, T2_SEEDS

    assert not set(T4_SEEDS) & set(T2_SEEDS) and len(T4_SEEDS) == 10
    for name in ("augment", "no_augment"):
        extra = {k: v for k, v in T4_ARMS[name].items() if T2_ARMS[name].get(k) != v}
        assert extra == {"buffer_unique": True, "settle_epochs": T4_SETTLE_EPOCHS,
                         "settle_lr_final": T4_SETTLE_LR_FINAL, "record_self_agreement": True}
        assert set(T2_ARMS[name]) <= set(T4_ARMS[name])
    assert T4_PASSES == T4_ITERATIONS + 1


@pytest.mark.parametrize("perfect,verdict", [(10, "supported"), (8, "supported"), (7, "inconclusive"),
                                             (6, "inconclusive"), (5, "refuted"), (0, "refuted")])
def test_each_arm_is_supported_at_8_of_10_refuted_at_5_inconclusive_between(perfect, verdict):
    r = settle_report(_both(perfect_aug=perfect, perfect_raw=10 - perfect if perfect <= 5 else perfect))
    assert r["augment"]["verdict"] == verdict and r["augment"]["perfect"] == perfect and r["integrity"] == []


def test_the_arms_are_judged_separately():
    r = settle_report(_both(perfect_aug=10, perfect_raw=2))
    assert r["augment"]["verdict"] == "supported" and r["no_augment"]["verdict"] == "refuted"


def test_the_verdict_reads_the_pass_AFTER_settling_not_the_best_or_the_last_iteration():
    arms = _both()
    for i, final in enumerate((2, 1, 1)):
        arms["augment"]["seeds"][i]["strict_failures_per_pass"] = [0] * (T4_PASSES - 1) + [final]
        arms["augment"]["seeds"][i]["final_failing_positions"] = list(range(final))
    assert settle_report(arms)["augment"]["perfect"] == 7
    arms = _both(perfect_aug=0)
    for row in arms["augment"]["seeds"]:
        row["strict_failures_per_pass"] = [4] * (T4_PASSES - 1) + [0]
        row["final_failing_positions"] = []
    r = settle_report(arms)["augment"]
    assert r["perfect"] == 10 and r["descriptives"]["perfect_before_settling"] == 0


def _set(path, value):
    def apply(arms):
        target = arms
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("augment", "training_fingerprint"), "27933b3a3bba"),
    "measurement": _set(("no_augment", "measurement_fingerprint"), "0" * 12),
    "fewer iterations": _set(("augment", "config", "iterations"), 6),
    "augment flipped": _set(("no_augment", "config", "augment"), True),
    "labels": _set(("augment", "config", "reanalyze_sims"), 32),
    "siblings off": _set(("augment", "config", "reanalyze_siblings"), False),
    "fifo buffer": _set(("augment", "config", "buffer_unique"), False),
    "no settling": _set(("no_augment", "config", "settle_epochs"), 0),
    "settle floor": _set(("augment", "config", "settle_lr_final"), 1e-3),
    "seeds": _set(("augment", "config", "seeds"), list(range(301, 311))),
    "a seed short of passes": _set(("augment", "seeds", 3, "strict_failures_per_pass"), [0] * T4_ITERATIONS),
    "scored on fewer positions": _set(("no_augment", "seeds", 2, "positions"), 627),
    "failure list disagrees": _set(("augment", "seeds", 4, "final_failing_positions"), [1, 2]),
    "siblings missing": _set(("augment", "seeds", 5, "history", 3, "siblings"), 0),
    "buffer not unique in the run": lambda arms: arms["augment"]["seeds"][6]["history"][7].pop("merged"),
    "settle pass missing": _set(("no_augment", "seeds", 1, "history"), _history()[:-1] + [_history()[-2]]),
    "settle not labelled": _set(("augment", "seeds", 0, "history", T4_PASSES - 1, "iteration"), T4_PASSES),
    "settle with other epochs": _set(("augment", "seeds", 0, "history", T4_PASSES - 1, "epochs"), 6),
    "agreement not recorded": lambda arms: arms["augment"]["seeds"][2]["history"][4].pop("self_agreement"),
    "a game missing": _set(("augment", "seeds", 5, "games_per_pass"), [48] * (T4_ITERATIONS - 1) + [47, 0]),
    "games during settling": _set(("augment", "seeds", 5, "games_per_pass"), [48] * T4_PASSES),
    "arm missing": lambda arms: arms.pop("no_augment"),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_a_broken_integrity_condition_makes_every_verdict_NOT_RUN(name):
    arms = _both()
    BREAKAGES[name](arms)
    r = settle_report(arms)
    assert r["integrity"] and all(r[a]["verdict"] == "not_run" for a in T4_ARMS)
    assert not any("could not be read" in p for p in r["integrity"])


@pytest.mark.parametrize("junk", [None, {}, {"augment": None, "no_augment": {}}])
def test_unreadable_evidence_is_NOT_RUN_never_an_exception(junk):
    r = settle_report(junk)
    assert all(r[a]["verdict"] == "not_run" for a in T4_ARMS)


def test_descriptives_give_the_curve_first_zero_relapses_and_the_count_perfect_before_settling():
    arms = _both()
    arms["augment"]["seeds"][0]["strict_failures_per_pass"] = [5] * 10 + [0] * (T4_PASSES - 10)
    d = settle_report(arms)["augment"]["descriptives"]
    assert len(d["mean_failures_per_pass"]) == T4_PASSES
    assert d["first_zero_pass"][0] == 11 and d["relapsed_seeds"] == [] and d["perfect_before_settling"] == 1
    arms["augment"]["seeds"][1]["strict_failures_per_pass"] = [0] * 5 + [2] * 5 + [0] * (T4_PASSES - 10)
    arms["augment"]["seeds"][2]["strict_failures_per_pass"] = [0, 1] + [0] * (T4_PASSES - 2)
    d = settle_report(arms)["augment"]["descriptives"]
    assert d["relapsed_seeds"] == [T4_SEEDS[1], T4_SEEDS[2]] and d["perfect_before_settling"] == 3


def test_self_agreement_is_read_against_the_truth_a_full_agreement_on_a_failing_pass_is_a_false_stop():
    arms = _both(perfect_aug=10)
    row = arms["augment"]["seeds"][0]
    row["strict_failures_per_pass"] = [3] * 20 + [0] * (T4_PASSES - 20)
    for i, h in enumerate(row["history"]):
        h["self_agreement"] = 1.0 if i >= 15 else 0.9
    arms["augment"]["seeds"][1]["history"] = _history(agreement=0.99)
    arms["augment"]["seeds"][1]["history"][-1]["self_agreement"] = 0.98
    d = settle_report(arms)["augment"]["descriptives"]["self_agreement"]
    assert d["false_stops"] == 5 + sum(1 for r in arms["augment"]["seeds"][2:] for f in r["strict_failures_per_pass"]
                                       if f > 0)
    assert d["final"][0] == 1.0 and d["final"][1] == 0.98
    assert d["first_full_pass"][0] == 16 and d["first_full_pass"][1] is None


def test_seed_rows_other_than_the_registered_seeds_are_NOT_RUN_even_when_the_config_names_the_right_ones():
    arms = _both()
    arms["augment"]["seeds"][0]["seed"] = 999
    r = settle_report(arms)
    assert r["augment"]["verdict"] == "not_run" and any("seeds" in p for p in r["integrity"])


def test_a_record_that_passes_integrity_but_cannot_be_judged_is_NOT_RUN_with_the_reason():
    arms = _both()
    arms["no_augment"]["seeds"][2]["strict_failures_per_pass"][3] = None
    r = settle_report(arms)
    assert r["no_augment"]["verdict"] == "not_run" and "could not be read" in r["integrity"][0]
