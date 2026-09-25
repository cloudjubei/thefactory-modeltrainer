"""Direct tests for harness/floor_converge.py — the pre-registered §C.48 T2 verdicts: does the generic self-play
process, run to convergence, give a RAW net optimal at every raw tic-tac-toe position (no averaging, no canonical
images)? Synthetic arms vary how many seeds end perfect and which integrity condition breaks."""
from __future__ import annotations

import pytest

from harness.floor_converge import (T2_ARMS, T2_ERA, T2_ITERATIONS, T2_MEASUREMENT_FP, T2_POSITIONS, T2_SEEDS,
                                    converge_report)


def _row(seed, final_failures=0, passes=T2_ITERATIONS, positions=T2_POSITIONS):
    curve = [max(0, final_failures + (passes - i) * 3) for i in range(1, passes + 1)]
    curve[-1] = final_failures
    return {"seed": seed, "params": 57453, "positions": positions, "strict_failures_per_pass": curve,
            "final_failing_positions": list(range(final_failures)), "games_per_pass": [48] * passes,
            "history": [{"iteration": i + 1, "siblings": 0 if i == 0 else 200} for i in range(passes)]}


def _arm(name, perfect=10, seeds=T2_SEEDS, **config):
    rows = [_row(s, final_failures=0 if i < perfect else 1 + i % 4) for i, s in enumerate(seeds)]
    return {"training_fingerprint": T2_ERA, "measurement_fingerprint": T2_MEASUREMENT_FP,
            "config": {**T2_ARMS[name], **config, "seeds": list(seeds)}, "seeds": rows}


def _both(perfect_aug=10, perfect_raw=10):
    return {"augment": _arm("augment", perfect_aug), "no_augment": _arm("no_augment", perfect_raw)}


@pytest.mark.parametrize("perfect,verdict", [(10, "supported"), (8, "supported"), (7, "inconclusive"),
                                             (6, "inconclusive"), (5, "refuted"), (0, "refuted")])
def test_each_arm_is_supported_at_8_of_10_refuted_at_5_inconclusive_between(perfect, verdict):
    r = converge_report(_both(perfect_aug=perfect, perfect_raw=10 - perfect if perfect <= 5 else perfect))
    assert r["augment"]["verdict"] == verdict and r["augment"]["perfect"] == perfect and r["integrity"] == []


def test_the_arms_are_judged_separately():
    r = converge_report(_both(perfect_aug=10, perfect_raw=2))
    assert r["augment"]["verdict"] == "supported" and r["no_augment"]["verdict"] == "refuted"


def test_the_verdict_reads_the_FINAL_pass_not_the_best_pass():
    arms = _both()
    for i, final in enumerate((2, 1, 1)):
        arms["augment"]["seeds"][i]["strict_failures_per_pass"] = [0] * (T2_ITERATIONS - 1) + [final]
        arms["augment"]["seeds"][i]["final_failing_positions"] = list(range(final))
    assert converge_report(arms)["augment"]["perfect"] == 7


def _set(path, value):
    def apply(arms):
        target = arms
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("augment", "training_fingerprint"), "2600dc4f574a"),
    "measurement": _set(("no_augment", "measurement_fingerprint"), "0" * 12),
    "fewer iterations": _set(("augment", "config", "iterations"), 6),
    "augment flipped": _set(("no_augment", "config", "augment"), True),
    "labels": _set(("augment", "config", "reanalyze_sims"), 32),
    "siblings off": _set(("augment", "config", "reanalyze_siblings"), False),
    "seeds": _set(("augment", "config", "seeds"), list(range(1, 11))),
    "a seed short of passes": _set(("augment", "seeds", 3, "strict_failures_per_pass"), [0] * (T2_ITERATIONS - 1)),
    "scored on fewer positions": _set(("no_augment", "seeds", 2, "positions"), 627),
    "failure list disagrees": _set(("augment", "seeds", 4, "final_failing_positions"), [1, 2]),
    "siblings missing": _set(("augment", "seeds", 5, "history", 3, "siblings"), 0),
    "a game missing": _set(("augment", "seeds", 5, "games_per_pass"), [48] * (T2_ITERATIONS - 1) + [47]),
    "arm missing": lambda arms: arms.pop("no_augment"),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_a_broken_integrity_condition_makes_every_verdict_NOT_RUN(name):
    arms = _both()
    BREAKAGES[name](arms)
    r = converge_report(arms)
    assert r["integrity"] and all(r[a]["verdict"] == "not_run" for a in T2_ARMS)
    assert not any("could not be read" in p for p in r["integrity"])


@pytest.mark.parametrize("junk", [None, {}, {"augment": None, "no_augment": {}}])
def test_unreadable_evidence_is_NOT_RUN_never_an_exception(junk):
    r = converge_report(junk)
    assert all(r[a]["verdict"] == "not_run" for a in T2_ARMS)


def test_descriptives_give_the_per_pass_mean_curve_and_when_each_seed_first_reached_zero():
    arms = _both()
    arms["augment"]["seeds"][0]["strict_failures_per_pass"] = [5] * 10 + [0] * (T2_ITERATIONS - 10)
    d = converge_report(arms)["augment"]["descriptives"]
    assert len(d["mean_failures_per_pass"]) == T2_ITERATIONS
    assert d["first_zero_pass"][0] == 11 and d["relapsed_seeds"] == []
    arms["augment"]["seeds"][1]["strict_failures_per_pass"] = [0] * 5 + [2] * 5 + [0] * (T2_ITERATIONS - 10)
    assert converge_report(arms)["augment"]["descriptives"]["relapsed_seeds"] == [T2_SEEDS[1]]


def test_seed_rows_other_than_the_registered_seeds_are_NOT_RUN_even_when_the_config_names_the_right_ones():
    arms = _both()
    arms["augment"]["seeds"][0]["seed"] = 999
    r = converge_report(arms)
    assert r["augment"]["verdict"] == "not_run" and any("seeds" in p for p in r["integrity"])


def test_a_record_that_passes_integrity_but_cannot_be_judged_is_NOT_RUN_with_the_reason():
    arms = _both()
    arms["no_augment"]["seeds"][2]["strict_failures_per_pass"][3] = None
    r = converge_report(arms)
    assert r["no_augment"]["verdict"] == "not_run" and "could not be read" in r["integrity"][0]
