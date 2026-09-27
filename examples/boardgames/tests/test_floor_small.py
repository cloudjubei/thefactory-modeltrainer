"""Direct tests for harness/floor_small.py — the pre-registered §C.49 T6 verdicts: does the working tic-tac-toe
process (T5's unique buffer, 2-deep siblings, settle) give a raw-perfect net at the ORACLE-FRONTIER setups instead of
the 57K net? Synthetic arms vary how many seeds end perfect and which integrity condition breaks."""
from __future__ import annotations

import pytest

from harness.floor_small import T6_ARMS, T6_ERA, T6_ITERATIONS, T6_MEASUREMENT_FP, T6_PARAMS, T6_SEEDS, small_report


def _row(name, seed, final=0):
    cfg = T6_ARMS[name]
    passes = T6_ITERATIONS + 1
    curve = [final + (passes - i) for i in range(1, passes + 1)]
    curve[-1] = final
    history = [{"iteration": i + 1, "siblings": 0 if i == 0 else 100, "merged": 50, "sibling_rings": [] if i == 0
                else [60, 40]} for i in range(T6_ITERATIONS)]
    history.append({"iteration": "settle", "epochs": cfg["settle_epochs"], "lr_final": cfg["settle_lr_final"]})
    return {"seed": seed, "params": T6_PARAMS[name], "positions": 4520, "strict_failures_per_pass": curve,
            "final_failing_positions": list(range(final)), "games_per_pass": [cfg["selfplay"]] * T6_ITERATIONS + [0],
            "history": history}


def _arm(name, perfect=10):
    return {"training_fingerprint": T6_ERA, "measurement_fingerprint": T6_MEASUREMENT_FP,
            "config": {**T6_ARMS[name], "seeds": list(T6_SEEDS)},
            "seeds": [_row(name, s, 0 if i < perfect else 1 + i % 3) for i, s in enumerate(T6_SEEDS)]}


def _all(**perfect):
    return {name: _arm(name, perfect.get(name, 10)) for name in T6_ARMS}


def test_the_arms_are_the_T5_process_at_the_frontier_setups_with_and_without_a_long_settle():
    from harness.floor_coverage import T5_ARMS

    base = T5_ARMS["augment_sib2"]
    assert set(T6_ARMS) == {"canon_mlp32", "canon_mlp32_long", "canon_conv6_long", "residual15", "residual15_long"}
    for name, cfg in T6_ARMS.items():
        canonical = name.startswith("canon")
        differs = {k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
        assert differs <= {"arch", "params", "augment", "settle_epochs"}
        assert cfg["augment"] is not canonical and cfg["arch"].get("canonical_input", False) is canonical
        assert cfg["settle_epochs"] == (1500 if name.endswith("_long") else base["settle_epochs"])
    assert T6_PARAMS == {"canon_mlp32": 938, "canon_mlp32_long": 938, "canon_conv6_long": 994, "residual15": 5008,
                         "residual15_long": 5008}
    assert not set(T6_SEEDS) & set(range(301, 321))


def test_the_registered_arches_build_nets_of_the_registered_sizes():
    from games.tictactoe import TicTacToe
    from harness.neural import Connect4Net, arch_for_game

    for name, cfg in T6_ARMS.items():
        net = Connect4Net(**arch_for_game(cfg["arch"], TicTacToe()))
        assert sum(p.numel() for p in net.parameters()) == T6_PARAMS[name]


@pytest.mark.parametrize("perfect,verdict", [(10, "supported"), (8, "supported"), (7, "inconclusive"),
                                             (6, "inconclusive"), (5, "refuted"), (0, "refuted")])
def test_each_arm_is_supported_at_8_of_10_refuted_at_5_inconclusive_between(perfect, verdict):
    r = small_report(_all(canon_conv6_long=perfect))
    assert r["integrity"] == [] and r["canon_conv6_long"]["verdict"] == verdict
    assert r["canon_conv6_long"]["perfect"] == perfect and r["residual15"]["verdict"] == "supported"


def test_the_verdict_reads_the_pass_after_the_settle():
    arms = _all()
    row = arms["canon_mlp32"]["seeds"][0]
    row["strict_failures_per_pass"][-1] = 3
    row["final_failing_positions"] = [1, 2, 3]
    assert small_report(arms)["canon_mlp32"]["perfect"] == 9
    arms = _all(canon_mlp32=0)
    for row in arms["canon_mlp32"]["seeds"]:
        row["strict_failures_per_pass"] = [5] * T6_ITERATIONS + [0]
        row["final_failing_positions"] = []
    r = small_report(arms)["canon_mlp32"]
    assert r["perfect"] == 10 and r["descriptives"]["perfect_before_settling"] == 0


def _set(path, value):
    def apply(arms):
        target = arms
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("canon_mlp32", "training_fingerprint"), "f1ee57d6152b"),
    "measurement": _set(("residual15", "measurement_fingerprint"), "0" * 12),
    "config": _set(("canon_conv6_long", "config", "settle_epochs"), 30),
    "seeds": _set(("residual15_long", "config", "seeds"), list(range(311, 321))),
    "seed rows": _set(("canon_mlp32", "seeds", 0, "seed"), 999),
    "params": _set(("canon_mlp32_long", "seeds", 2, "params"), 57453),
    "passes": _set(("residual15", "seeds", 3, "strict_failures_per_pass"), [0] * T6_ITERATIONS),
    "positions": _set(("canon_conv6_long", "seeds", 4, "positions"), 627),
    "failure list": _set(("canon_mlp32", "seeds", 5, "final_failing_positions"), [1]),
    "games": _set(("residual15_long", "seeds", 6, "games_per_pass"), [48] * (T6_ITERATIONS + 1)),
    "settle epochs in the run": _set(("canon_mlp32_long", "seeds", 7, "history", T6_ITERATIONS, "epochs"), 30),
    "no settle in the run": _set(("canon_mlp32", "seeds", 8, "history", T6_ITERATIONS, "iteration"), 31),
    "one ring in the run": _set(("residual15", "seeds", 9, "history", 4, "sibling_rings"), [100]),
    "arm missing": lambda arms: arms.pop("canon_conv6_long"),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_a_broken_integrity_condition_makes_every_verdict_NOT_RUN(name):
    arms = _all()
    BREAKAGES[name](arms)
    r = small_report(arms)
    assert r["integrity"] and all(r[a]["verdict"] == "not_run" for a in T6_ARMS)
    assert not any("could not be read" in p for p in r["integrity"])


@pytest.mark.parametrize("junk", [None, {}, {"canon_mlp32": None}])
def test_unreadable_evidence_is_NOT_RUN_never_an_exception(junk):
    assert all(small_report(junk)[a]["verdict"] == "not_run" for a in T6_ARMS)


def test_a_record_that_cannot_be_judged_is_NOT_RUN_with_the_reason():
    arms = _all()
    arms["residual15"]["seeds"][0]["history"][-1] = None
    r = small_report(arms)
    assert r["residual15"]["verdict"] == "not_run" and "could not be read" in r["integrity"][0]
