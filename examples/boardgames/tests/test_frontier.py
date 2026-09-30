"""Direct tests for harness/frontier.py — the oracle-frontier process: families of nets with one width parameter, a
target of positions with exact answers, a fit to it, and the smallest-width search over seeds."""
from __future__ import annotations

import pytest

from harness.frontier import (DEFAULT_RECIPE, arch_at, enumerated_target, failures, family_name, fit,
                              frontier_summary, probe_width, smallest_width)


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
    r = probe_width(ttt, {"body": "mlp"}, 4, targets[False], [1, 2, 3], DEFAULT_RECIPE, stop_on_failure=True)
    assert not r["ok"] and calls == [1, 2] and r["params"] == 10
    calls.clear()
    r = probe_width(ttt, {"body": "mlp"}, 4, targets[False], [1, 2, 3], DEFAULT_RECIPE)
    assert not r["ok"] and calls == [1, 2, 3] and r["rate"] == pytest.approx(2 / 3)
    calls.clear()
    outcomes[2] = True
    r = probe_width(ttt, {"body": "mlp"}, 4, targets[False], [1, 2, 3], DEFAULT_RECIPE)
    assert r["ok"] and calls == [1, 2, 3] and r["rate"] == 1.0


def test_patience_counts_from_the_first_epoch_that_reached_the_best_not_from_later_ties(ttt, targets, monkeypatch):
    import harness.frontier as frontier

    script = iter([9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    monkeypatch.setattr(frontier, "failures", lambda net, ev: next(script))
    recipe = {**DEFAULT_RECIPE, "max_epochs": 1000, "check_every": 10, "patience": 50}
    r = fit(ttt, arch_at({"body": "mlp"}, 2), targets[False], 1, recipe)
    assert r["best_failures"] == 4 and r["epochs_run"] == 20 + 50 and not r["solved"]


def _probe(width, rate):
    return {"width": width, "params": width * 10, "ok": rate == 1.0, "rate": rate}


def test_the_summary_separates_the_strict_the_robust_and_the_first_success_frontier():
    probes = [_probe(8, 0.0), _probe(24, 0.4), _probe(28, 0.6), _probe(29, 1.0), _probe(30, 1.0), _probe(31, 0.8),
              _probe(32, 1.0), _probe(64, 1.0), _probe(16, 0.0)]
    s = frontier_summary(probes, strict=29)
    assert s["strict"] == {"width": 29, "params": 290}
    assert s["robust"] == {"width": 32, "params": 320}
    assert s["first_success"] == {"width": 24, "params": 240}
    assert s["rates"] == [[8, 80, 0.0], [16, 160, 0.0], [24, 240, 0.4], [28, 280, 0.6], [29, 290, 1.0],
                          [30, 300, 1.0], [31, 310, 0.8], [32, 320, 1.0], [64, 640, 1.0]]


def test_a_search_with_no_success_has_no_frontier_of_any_kind():
    s = frontier_summary([_probe(2, 0.0), _probe(4, 0.0)], strict=None)
    assert s["strict"] is None and s["robust"] is None and s["first_success"] is None


def test_a_widest_probe_that_fails_leaves_no_robust_frontier():
    s = frontier_summary([_probe(8, 1.0), _probe(16, 0.8)], strict=8)
    assert s["robust"] is None and s["strict"] == {"width": 8, "params": 80}


def test_a_fit_hands_back_its_net_only_when_asked(ttt, targets):
    recipe = {**DEFAULT_RECIPE, "max_epochs": 20, "check_every": 10}
    plain = fit(ttt, arch_at({"body": "mlp"}, 4), targets[False], 3, recipe)
    with_net = fit(ttt, arch_at({"body": "mlp"}, 4), targets[False], 3, recipe, return_net=True)
    assert "net" not in plain and failures(with_net.pop("net"), targets[False]["eval"]) == with_net["curve"][-1][1]
    assert with_net == plain


def test_a_fit_can_continue_from_a_given_net_instead_of_a_fresh_one(ttt, targets):
    import copy

    import torch

    recipe = {**DEFAULT_RECIPE, "max_epochs": 3000}
    held = fit(ttt, arch_at({"body": "mlp", "canonical": True}, 32), targets[True], 1, recipe, return_net=True)
    assert held["solved"]
    start = copy.deepcopy(held["net"])
    again = fit(ttt, arch_at({"body": "mlp", "canonical": True}, 32), targets[True], 2, recipe, return_net=True,
                init_net=held["net"])
    assert again["solved"] and again["solved_at_epoch"] == recipe["check_every"]
    assert again["net"] is held["net"]
    fresh = fit(ttt, arch_at({"body": "mlp", "canonical": True}, 32), targets[True], 2,
                {**recipe, "max_epochs": 25, "check_every": 25})
    assert fresh["curve"][0][1] > 0
    assert not all(torch.equal(a, b) for a, b in zip(start.parameters(), again["net"].parameters()))


def test_a_given_net_of_another_shape_is_refused(ttt, targets):
    from harness.neural import Connect4Net, arch_for_game

    other = Connect4Net(**arch_for_game({"mlp_hidden": [8]}, ttt))
    with pytest.raises(ValueError, match="arch"):
        fit(ttt, arch_at({"body": "mlp"}, 16), targets[False], 1, DEFAULT_RECIPE, init_net=other)


def test_a_recipe_can_decay_the_learning_rate_linearly_to_its_floor_over_the_epoch_budget(ttt, targets, monkeypatch):
    import torch

    seen = []
    real = torch.optim.Adam.step

    def step(self, *a, **k):
        seen.append(self.param_groups[0]["lr"])
        return real(self, *a, **k)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    recipe = {**DEFAULT_RECIPE, "lr": 1e-3, "lr_end": 1e-4, "max_epochs": 10, "check_every": 5, "patience": None}
    fit(ttt, arch_at({"body": "mlp"}, 2), targets[True], 1, recipe)
    per_epoch = len(seen) // 10
    firsts = [seen[e * per_epoch] for e in range(10)]
    assert firsts == pytest.approx([1e-3 - 1e-4 * e for e in range(10)])
    assert seen[-1] == pytest.approx(1e-4)


def test_without_patience_an_unsolved_fit_uses_its_whole_budget(ttt, targets):
    recipe = {**DEFAULT_RECIPE, "max_epochs": 60, "check_every": 10, "patience": None}
    r = fit(ttt, arch_at({"body": "mlp"}, 1), targets[False], 1, recipe)
    assert not r["solved"] and r["epochs_run"] == 60


def test_a_recipe_without_the_new_keys_trains_exactly_as_before(ttt, targets):
    recipe = {**DEFAULT_RECIPE, "max_epochs": 40, "check_every": 10}
    a = fit(ttt, arch_at({"body": "mlp"}, 4), targets[False], 5, recipe)
    b = fit(ttt, arch_at({"body": "mlp"}, 4), targets[False], 5, {**recipe, "lr_end": None})
    assert a == b
