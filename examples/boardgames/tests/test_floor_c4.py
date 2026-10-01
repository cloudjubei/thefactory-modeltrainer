"""Direct tests for harness/floor_c4.py — T10: the solver-free Connect-4 run is judged by the solver on the net it
stopped at. A seed succeeds only when its run stopped on full agreement AND the solver certifies that net's own
first-player tree through the registered depth."""
from __future__ import annotations

import pytest

from harness.floor_c4 import c4_report

SPEC = {"config": {"selfplay": 4, "iterations": 6, "settle_epochs": 3, "x": 1},
        "seeds": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), "era": "e" * 12, "measurement_fp": "m" * 12, "params": 100, "certify_depth": 10, "support_at": 8,
        "refute_at": 5}


def _row(seed, stop_at=4, certified=True):
    cfg = SPEC["config"]
    history = [{"iteration": 1, "siblings": 0, "tree_walked": 0, "tree_positions": 0, "tree_disagreements": 0}]
    last = stop_at if stop_at else cfg["iterations"]
    history += [{"iteration": i, "siblings": 0, "tree_walked": 9, "tree_positions": 2, "tree_disagreements": 5}
                for i in range(2, last + 1)]
    if stop_at:
        history[-1] = {**history[-1], "tree_disagreements": 0, "stopped": True}
        games = [cfg["selfplay"]] * (stop_at - 1)
    else:
        history.append({"iteration": "settle", "epochs": cfg["settle_epochs"]})
        games = [cfg["selfplay"]] * cfg["iterations"] + [0]
    return {"seed": seed, "stopped": bool(stop_at), "params": SPEC["params"], "games_per_pass": games,
            "net": {"path": f"n{seed}.pt", "weights_sha": "w", "file_sha256": f"f{seed}"}, "history": history,
            "certificate": {"certified": certified, "horizon": SPEC["certify_depth"], "failures": 0 if certified else 3,
                            "net_file_sha256": f"f{seed}", "values": {"recorded": 5, "solved": 2}}}


def _evidence(rows=None):
    rows = rows or {}
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": {**SPEC["config"], "seeds": list(SPEC["seeds"]), "certify_depth": SPEC["certify_depth"]},
            "seeds": [_row(s, **rows.get(s, {})) for s in SPEC["seeds"]]}


def test_every_seed_stopping_on_a_certified_net_is_supported():
    r = c4_report(_evidence(), SPEC)
    assert r["integrity"] == [] and r["verdict"] == "supported" and r["successes"] == 10
    assert r["descriptives"]["stopped"] == 10 and r["descriptives"]["stop_iterations"] == [4] * 10


@pytest.mark.parametrize("failing,verdict", [(0, "supported"), (2, "supported"), (3, "inconclusive"),
                                             (4, "inconclusive"), (5, "refuted"), (10, "refuted")])
def test_seeds_whose_stopped_net_the_solver_rejects_count_against_the_bars(failing, verdict):
    r = c4_report(_evidence({s: {"certified": False} for s in SPEC["seeds"][:failing]}), SPEC)
    assert r["successes"] == 10 - failing and r["verdict"] == verdict
    if failing:
        assert r["descriptives"]["false_stops"] == failing


def test_a_seed_that_never_stopped_fails_even_if_its_final_net_is_certified():
    r = c4_report(_evidence({s: {"stop_at": None} for s in SPEC["seeds"][:3]}), SPEC)
    assert r["successes"] == 7 and r["verdict"] == "inconclusive"
    assert r["descriptives"]["certified_without_stopping"] == 3


def _set(path, value):
    def apply(e):
        target = e
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("training_fingerprint",), "0" * 12),
    "measurement": _set(("measurement_fingerprint",), "0" * 12),
    "config": _set(("config", "x"), 2),
    "certify depth": _set(("config", "certify_depth"), 8),
    "seeds": _set(("config", "seeds"), [1, 2]),
    "seed rows": _set(("seeds", 0, "seed"), 99),
    "params": _set(("seeds", 1, "params"), 7),
    "another net certified": _set(("seeds", 2, "certificate", "net_file_sha256"), "other"),
    "shallower certificate": _set(("seeds", 3, "certificate", "horizon"), 8),
    "certificate verdict": _set(("seeds", 3, "certificate", "certified"), 1),
    "stop flag without a stop": _set(("seeds", 4, "stopped"), False),
    "stopped on disagreement": _set(("seeds", 5, "history", 3, "tree_disagreements"), 2),
    "stopped before any walk": _set(("seeds", 6, "history"), [{"iteration": 1, "stopped": True, "tree_walked": 0,
                                                             "tree_disagreements": 0, "siblings": 0}]),
    "siblings came back": _set(("seeds", 7, "history", 2, "siblings"), 4),
    "games": _set(("seeds", 8, "games_per_pass"), [4] * 9),
    "unstopped without the full run": lambda e: e["seeds"].__setitem__(9, {**_row(10, stop_at=None), "history": (
        _row(10, stop_at=None)["history"][:3] + _row(10, stop_at=None)["history"][-1:])}),
    "no walk on an iteration": _set(("seeds", 0, "history", 2, "tree_walked"), 0),
}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_evidence_that_is_not_the_registered_run_is_not_run(breakage):
    e = _evidence()
    BREAKAGES[breakage](e)
    r = c4_report(e, SPEC)
    assert r["integrity"] and r["verdict"] == "not_run"


@pytest.mark.parametrize("e", [None, {}, {"seeds": "x"}, []])
def test_missing_or_unreadable_evidence_is_not_run(e):
    assert c4_report(e, SPEC)["verdict"] == "not_run"
