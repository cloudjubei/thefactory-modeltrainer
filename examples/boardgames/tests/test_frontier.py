"""Direct tests for harness/frontier.py — the oracle-frontier process: families of nets with one width parameter, a
target of positions with exact answers, a fit to it, and the smallest-width search over seeds."""
from __future__ import annotations

import pytest

from harness.frontier import (DEFAULT_RECIPE, arch_at, enumerated_target, failures, family_name, fit, probe_width,
                              smallest_width)


@pytest.mark.parametrize("body,arch", [("mlp", {"mlp_hidden": [7]}), ("mlp2", {"mlp_hidden": [7, 7]}),
                                       ("conv", {"channels": 7}),
                                       ("residual", {"channels": 7, "blocks": 1, "head_hidden": 7, "residual": True})])
def test_each_family_is_one_shape_with_one_width(body, arch):
    assert arch_at({"body": body}, 7) == arch
    assert arch_at({"body": body, "canonical": True}, 7) == {**arch, "canonical_input": True}
    assert family_name({"body": body, "canonical": True}) == f"canon_{body}" and family_name({"body": body}) == body


@pytest.mark.parametrize("family,width", [({"body": "lstm"}, 4), ({"body": "mlp"}, 0)])
def test_an_unknown_body_or_a_width_below_one_is_refused(family, width):
    with pytest.raises(ValueError, match="width"):
        arch_at(family, width)


def _threshold(t, fails=()):
    seen = []

    def probe(w):
        seen.append(w)
        return w >= t and w not in fails
    return probe, seen


@pytest.mark.parametrize("t", [1, 2, 3, 5, 17, 64, 100, 1000])
def test_the_search_finds_the_smallest_succeeding_width_of_a_monotone_family(t):
    probe, seen = _threshold(t)
    r = smallest_width(probe, start=2, cap=1024, confirm_above=2)
    assert r["frontier"] == t and r["non_monotone"] == []
    assert r["probes"][t] and (t == 1 or not r["probes"][t - 1])
    assert len(seen) == len(set(seen)) and len(seen) <= 2 * 11 + 2


def test_nothing_up_to_the_cap_succeeding_gives_no_frontier():
    probe, seen = _threshold(5000)
    r = smallest_width(probe, start=2, cap=100)
    assert r["frontier"] is None and max(seen) == 100 and not any(r["probes"].values())


def test_a_failure_just_above_the_frontier_is_flagged_not_hidden():
    probe, _ = _threshold(12, fails=(13,))
    r = smallest_width(probe, start=2, cap=1024, confirm_above=2)
    assert r["frontier"] == 12 and r["non_monotone"] == [13] and r["probes"][14]
    assert smallest_width(_threshold(12, fails=(13,))[0], confirm_above=0)["non_monotone"] == []


@pytest.mark.parametrize("start,cap", [(0, 10), (8, 4)])
def test_a_search_range_that_is_empty_is_refused(start, cap):
    with pytest.raises(ValueError, match="start"):
        smallest_width(lambda w: True, start=start, cap=cap)


@pytest.fixture(scope="module")
def ttt():
    from games.tictactoe import TicTacToe

    return TicTacToe()


@pytest.fixture(scope="module")
def targets(ttt):
    return {c: enumerated_target(ttt, c) for c in (False, True)}


def test_the_enumerated_target_grades_every_raw_position_and_trains_once_per_input_the_net_can_tell_apart(ttt, targets):
    from harness.coverage import optimal_actions, reachable_states

    raw = [s for s in reachable_states(ttt, exact=True, symmetry=False)[0] if not ttt.is_terminal(s)]
    for canonical, rows in ((False, 4520), (True, 627)):
        t = targets[canonical]
        assert t["positions"] == len(raw) == 4520 and len(t["eval"]["x"]) == 4520
        assert len(t["train"]["x"]) == rows == len(t["train"]["value"]) and t["allowed_failures"] == 0
        assert t["train"]["policy"].sum(dim=1).allclose(t["train"]["policy"].new_ones(rows))
    ev = targets[False]["eval"]
    for i in range(0, 4520, 97):
        assert sorted(ev["optimal"][i].nonzero().flatten().tolist()) == sorted(optimal_actions(ttt, raw[i]))
        assert sorted(ev["legal"][i].nonzero().flatten().tolist()) == sorted(ttt.legal_actions(raw[i]))


class _Fixed:
    training = True

    def __init__(self, logits):
        self.logits = logits

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def __call__(self, x):
        return self.logits, None


def test_failures_count_positions_whose_legal_argmax_is_not_optimal(targets):
    ev = targets[False]["eval"]
    perfect = ev["optimal"].float() - 5 * (~ev["legal"]).float()
    assert failures(_Fixed(perfect), ev) == 0
    wrong = perfect.clone()
    planted = [i for i in range(len(ev["x"])) if (ev["legal"][i] & ~ev["optimal"][i]).any()][:: 700][:4]
    assert len(planted) == 4
    for i in planted:
        bad = (ev["legal"][i] & ~ev["optimal"][i]).nonzero().flatten()
        wrong[i, bad[0]] = 9.0
    assert failures(_Fixed(wrong), ev) == len(planted)
    illegal = perfect.clone()
    illegal[:, :] += 50 * (~ev["legal"]).float()
    assert failures(_Fixed(illegal), ev) == 0


def test_a_fit_that_reaches_the_target_stops_and_reports_when(ttt, targets):
    r = fit(ttt, arch_at({"body": "mlp", "canonical": True}, 32), targets[True], 1, DEFAULT_RECIPE)
    assert r["solved"] and r["params"] == 938 and r["solved_at_epoch"] == r["epochs_run"] < DEFAULT_RECIPE["max_epochs"]
    assert r["curve"][-1] == [r["solved_at_epoch"], 0] and r["best_failures"] == 0


def test_a_fit_that_stops_improving_ends_at_its_patience(ttt, targets):
    recipe = {**DEFAULT_RECIPE, "max_epochs": 400, "check_every": 10, "patience": 50}
    r = fit(ttt, arch_at({"body": "mlp"}, 1), targets[False], 1, recipe)
    assert not r["solved"] and r["solved_at_epoch"] is None and r["epochs_run"] < 400
    best_epoch = min(e for e, f in r["curve"] if f == r["best_failures"])
    assert r["epochs_run"] - best_epoch >= 50 and r["epochs_run"] - best_epoch < 60


def test_the_same_seed_fits_the_same_net(ttt, targets):
    recipe = {**DEFAULT_RECIPE, "max_epochs": 30, "check_every": 10}
    a = fit(ttt, arch_at({"body": "mlp"}, 4), targets[False], 3, recipe)
    b = fit(ttt, arch_at({"body": "mlp"}, 4), targets[False], 3, recipe)
    c = fit(ttt, arch_at({"body": "mlp"}, 4), targets[False], 4, recipe)
    assert a["curve"] == b["curve"] and a["curve"] != c["curve"]


def test_a_width_succeeds_only_if_every_seed_does_and_the_first_failure_ends_the_probe(ttt, targets, monkeypatch):
    import harness.frontier as frontier

    outcomes = {1: True, 2: False, 3: True}
    calls = []

    def fake_fit(game, arch, target, seed, recipe):
        calls.append(seed)
        return {"seed": seed, "params": 10, "solved": outcomes[seed]}
    monkeypatch.setattr(frontier, "fit", fake_fit)
    r = probe_width(ttt, {"body": "mlp"}, 4, targets[False], [1, 2, 3], DEFAULT_RECIPE)
    assert not r["ok"] and calls == [1, 2] and r["params"] == 10
    calls.clear()
    outcomes[2] = True
    assert probe_width(ttt, {"body": "mlp"}, 4, targets[False], [1, 2, 3], DEFAULT_RECIPE)["ok"] and calls == [1, 2, 3]


def test_patience_counts_from_the_first_epoch_that_reached_the_best_not_from_later_ties(ttt, targets, monkeypatch):
    import harness.frontier as frontier

    script = iter([9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    monkeypatch.setattr(frontier, "failures", lambda net, ev: next(script))
    recipe = {**DEFAULT_RECIPE, "max_epochs": 1000, "check_every": 10, "patience": 50}
    r = fit(ttt, arch_at({"body": "mlp"}, 2), targets[False], 1, recipe)
    assert r["best_failures"] == 4 and r["epochs_run"] == 20 + 50 and not r["solved"]
