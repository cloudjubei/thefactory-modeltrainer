"""Direct tests for harness/arm_judge.py — the generic pre-registered verdict for arms of a tic-tac-toe process run:
per arm, SUPPORTED when enough seeds end raw-perfect after the settle, REFUTED at few enough, INCONCLUSIVE between,
and NOT_RUN for every arm when the evidence is not the registered run."""
from __future__ import annotations

import pytest

from harness.arm_judge import report

SPEC = {"arms": {"a": {"selfplay": 4, "settle_epochs": 7, "sibling_depth": 2, "x": 1},
                 "b": {"selfplay": 4, "settle_epochs": 30, "sibling_depth": 1, "x": 2}},
        "params": {"a": 100, "b": 200}, "seeds": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "era": "e" * 12,
        "measurement_fp": "m" * 12, "iterations": 3, "positions": 50, "support_at": 8, "refute_at": 5}


def _row(name, seed, final=0):
    cfg = SPEC["arms"][name]
    passes = SPEC["iterations"] + 1
    curve = [final + (passes - i) for i in range(1, passes + 1)]
    curve[-1] = final
    history = [{"iteration": i + 1, "merged": 3, "sibling_rings": [] if i == 0 else [2] * cfg["sibling_depth"]}
               for i in range(SPEC["iterations"])]
    history.append({"iteration": "settle", "epochs": cfg["settle_epochs"]})
    return {"seed": seed, "params": SPEC["params"][name], "positions": SPEC["positions"],
            "strict_failures_per_pass": curve, "final_failing_positions": list(range(final)),
            "games_per_pass": [cfg["selfplay"]] * SPEC["iterations"] + [0], "history": history}


def _arm(name, perfect=10):
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": {**SPEC["arms"][name], "seeds": list(SPEC["seeds"])},
            "seeds": [_row(name, s, 0 if i < perfect else 1 + i % 3) for i, s in enumerate(SPEC["seeds"])]}


def _all(**perfect):
    return {n: _arm(n, perfect.get(n, 10)) for n in SPEC["arms"]}


@pytest.mark.parametrize("perfect,verdict", [(10, "supported"), (8, "supported"), (7, "inconclusive"),
                                             (6, "inconclusive"), (5, "refuted"), (0, "refuted")])
def test_each_arm_is_judged_on_its_own_against_the_bars(perfect, verdict):
    r = report(_all(a=perfect), SPEC)
    assert r["integrity"] == [] and r["a"]["verdict"] == verdict and r["a"]["perfect"] == perfect
    assert r["b"]["verdict"] == "supported"


def test_the_verdict_reads_the_pass_after_the_settle_and_reports_the_pass_before():
    arms = _all(a=0)
    for row in arms["a"]["seeds"]:
        row["strict_failures_per_pass"] = [4] * SPEC["iterations"] + [0]
        row["final_failing_positions"] = []
    r = report(arms, SPEC)["a"]
    assert r["perfect"] == 10 and r["descriptives"]["perfect_before_settling"] == 0
    assert r["descriptives"]["failures_before_settling"] == [4] * 10


def _set(path, value):
    def apply(arms):
        target = arms
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("a", "training_fingerprint"), "0" * 12),
    "measurement": _set(("b", "measurement_fingerprint"), "0" * 12),
    "config": _set(("a", "config", "x"), 9),
    "seeds": _set(("b", "config", "seeds"), [1, 2]),
    "seed rows": _set(("a", "seeds", 0, "seed"), 99),
    "params": _set(("b", "seeds", 2, "params"), 7),
    "passes": _set(("a", "seeds", 3, "strict_failures_per_pass"), [0] * 3),
    "positions": _set(("b", "seeds", 4, "positions"), 49),
    "failure list": _set(("a", "seeds", 5, "final_failing_positions"), [1]),
    "games": _set(("b", "seeds", 6, "games_per_pass"), [4] * 4),
    "settle epochs": _set(("a", "seeds", 7, "history", 3, "epochs"), 30),
    "no settle": _set(("b", "seeds", 8, "history", 3, "iteration"), 4),
    "rings": _set(("a", "seeds", 9, "history", 2, "sibling_rings"), [2]),
    "no unique buffer": lambda arms: arms["b"]["seeds"][1]["history"][1].pop("merged"),
    "arm missing": lambda arms: arms.pop("b"),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_a_broken_integrity_condition_makes_every_verdict_NOT_RUN(name):
    arms = _all()
    BREAKAGES[name](arms)
    r = report(arms, SPEC)
    assert r["integrity"] and all(r[a]["verdict"] == "not_run" for a in SPEC["arms"])
    assert not any("could not be read" in p for p in r["integrity"])


@pytest.mark.parametrize("junk", [None, {}, {"a": None}])
def test_unreadable_evidence_is_NOT_RUN_never_an_exception(junk):
    assert all(report(junk, SPEC)[a]["verdict"] == "not_run" for a in SPEC["arms"])


def test_a_record_that_cannot_be_read_is_NOT_RUN_with_the_reason():
    arms = _all()
    arms["b"]["seeds"][0]["history"][-1] = None
    r = report(arms, SPEC)
    assert r["b"]["verdict"] == "not_run" and "could not be read" in r["integrity"][0]
