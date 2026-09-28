"""Direct tests for harness/strategy_fit.py — the oracle frontier for a game too large to enumerate: grow the target
from the net's OWN strategy tree, round by round (walk the positions the net's moves reach against every reply,
label them with the exact solver, retrain), until the net is certified through the horizon or stops improving."""
from __future__ import annotations

import random

import pytest

from harness.strategy_fit import expand_round, fit_strategy, strategy_target


@pytest.fixture(scope="module")
def ttt():
    from games.tictactoe import TicTacToe

    return TicTacToe()


def _move_values(game):
    from harness.coverage import move_values

    calls = {"n": 0}

    def fn(s):
        calls["n"] += 1
        return move_values(game, s)
    fn.calls = calls
    return fn


def _optimal(game):
    from harness.coverage import optimal_actions

    return lambda states: [min(optimal_actions(game, s)) for s in states]


def _certify_tree(game, root, player, choose):
    from harness.certify import certify

    return certify(game, root, player, choose, lambda s: game.position_value(s))


@pytest.mark.parametrize("player", [0, 1])
def test_a_round_over_a_perfect_strategy_labels_every_player_position_and_finds_no_failure(ttt, player):
    from harness.coverage import optimal_actions

    root = ttt.initial_state(random.Random(0))
    r = expand_round(ttt, root, player, _optimal(ttt), _move_values(ttt), depth=None, known={})
    reference = _certify_tree(ttt, root, player, _optimal(ttt))
    assert r["failures"] == [] and r["complete"] and len(r["labelled"]) == reference["nodes"]["player"]
    for key, (state, best, value) in r["labelled"].items():
        assert key == ttt.state_key(state) and sorted(best) == sorted(optimal_actions(ttt, state))
        assert value == ttt.position_value(state)


def test_a_misplayed_position_is_a_failure_labelled_with_its_best_moves_and_not_followed(ttt):
    from harness.coverage import optimal_actions

    root = ttt.initial_state(random.Random(0))
    best = _optimal(ttt)
    wrong_at = ttt.step(root, 4, random.Random(0))
    wrong_key = ttt.state_key(wrong_at)
    bad = next(a for a in ttt.legal_actions(wrong_at) if a not in optimal_actions(ttt, wrong_at))

    def chooser(states):
        return [bad if ttt.state_key(s) == wrong_key else best([s])[0] for s in states]
    r = expand_round(ttt, root, 1, chooser, _move_values(ttt), depth=None, known={})
    assert r["failures"] == [wrong_key] and sorted(r["labelled"][wrong_key][1]) == sorted(optimal_actions(ttt, wrong_at))
    walked = _certify_tree(ttt, root, 1, chooser)
    assert len(r["labelled"]) == walked["nodes"]["player"]


def test_known_labels_are_reused_not_recomputed(ttt):
    root = ttt.initial_state(random.Random(0))
    fn = _move_values(ttt)
    known: dict = {}
    first = expand_round(ttt, root, 0, _optimal(ttt), fn, depth=None, known=known)
    solved = fn.calls["n"]
    again = expand_round(ttt, root, 0, _optimal(ttt), fn, depth=None, known=known)
    assert fn.calls["n"] == solved and again["labelled"].keys() == first["labelled"].keys() and len(known) == solved


def test_the_horizon_bounds_the_walk(ttt):
    root = ttt.initial_state(random.Random(0))
    r = expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=3, known={})
    full = expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=None, known={})
    assert not r["complete"] and 0 < len(r["labelled"]) < len(full["labelled"])
    assert all(sum(1 for v in s.board if v) < 3 for s, _b, _v in r["labelled"].values())


def test_the_target_teaches_every_best_move_equally_and_grades_by_them(ttt):
    root = ttt.initial_state(random.Random(0))
    r = expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=None, known={})
    t = strategy_target(ttt, r["labelled"])
    n = len(r["labelled"])
    assert len(t["train"]["x"]) == n == len(t["eval"]["x"]) and t["allowed_failures"] == 0
    for i, (_s, best, _v) in enumerate(r["labelled"].values()):
        assert sorted(t["eval"]["optimal"][i].nonzero().flatten().tolist()) == sorted(best)
        assert t["train"]["policy"][i].sum().item() == pytest.approx(1.0)


def test_a_big_enough_net_is_grown_to_a_certified_strategy_and_a_tiny_one_is_not(ttt):
    from harness.frontier import DEFAULT_RECIPE

    root = ttt.initial_state(random.Random(0))
    recipe = {**DEFAULT_RECIPE, "max_epochs": 3000}
    ok = fit_strategy(ttt, {"mlp_hidden": [64], "canonical_input": True}, root, 0, None, 1, recipe, 8,
                      _move_values(ttt), known={})
    assert ok["certified"] and ok["rounds"][-1]["failures"] == 0 and len(ok["rounds"]) >= 2
    assert ok["rounds"][0]["failures"] > 0
    tiny = fit_strategy(ttt, {"mlp_hidden": [1]}, root, 0, None, 1, {**recipe, "max_epochs": 200}, 3,
                        _move_values(ttt), known={})
    assert not tiny["certified"] and len(tiny["rounds"]) == 3


def test_a_batch_labeller_is_asked_once_per_ply_for_the_unknown_positions_only(ttt):
    from harness.coverage import move_values

    root = ttt.initial_state(random.Random(0))
    batches = []

    def many(states):
        batches.append(len(states))
        return [move_values(ttt, s) for s in states]
    known: dict = {}
    r = expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=None, known=known, move_values_many=many)
    plain = expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=None, known={})
    assert r["labelled"].keys() == plain["labelled"].keys() and sum(batches) == len(known)
    batches.clear()
    expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=None, known=known, move_values_many=many)
    assert batches == []


def test_a_horizon_of_two_plies_labels_only_the_first_player_position(ttt):
    root = ttt.initial_state(random.Random(0))
    r = expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=2, known={})
    assert [sum(1 for v in s.board if v) for s, _b, _v in r["labelled"].values()] == [0]


def test_every_round_trains_on_everything_labelled_so_far(ttt, monkeypatch):
    import harness.strategy_fit as sf
    from harness.frontier import DEFAULT_RECIPE

    seen = []
    real = sf.expand_round

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(set(out["labelled"]))
        return out
    monkeypatch.setattr(sf, "expand_round", spy)
    root = ttt.initial_state(random.Random(0))
    r = fit_strategy(ttt, {"mlp_hidden": [4]}, root, 0, None, 1, {**DEFAULT_RECIPE, "max_epochs": 50}, 4,
                     _move_values(ttt), known={})
    union = set()
    for entry, keys in zip(r["rounds"], seen, strict=True):
        union |= keys
        assert entry["positions"] == len(union) and entry["labelled"] == len(keys)
    assert len(union) > max(len(k) for k in seen)


def test_a_single_failure_is_never_certified(ttt, monkeypatch):
    import harness.strategy_fit as sf
    from harness.frontier import DEFAULT_RECIPE

    monkeypatch.setattr(sf, "expand_round", lambda *a, **k: {"labelled": {}, "failures": ["x"],
                                                             "failures_by_depth": {0: 1}, "complete": True,
                                                             "nodes": {}})
    monkeypatch.setattr(sf, "strategy_target", lambda game, data: None)
    import harness.frontier as frontier
    monkeypatch.setattr(frontier, "fit", lambda *a, **k: {"net": sf_net(ttt), "solved": False, "solved_at_epoch": None,
                                                          "best_failures": 1, "epochs_run": 1})
    root = ttt.initial_state(random.Random(0))
    r = fit_strategy(ttt, {"mlp_hidden": [4]}, root, 0, None, 1, DEFAULT_RECIPE, 3, _move_values(ttt), known={})
    assert not r["certified"] and len(r["rounds"]) == 3


def sf_net(game):
    from harness.neural import Connect4Net, arch_for_game

    return Connect4Net(**arch_for_game({"mlp_hidden": [4]}, game))


def _won_root(ttt):
    r = random.Random(0)
    s = ttt.initial_state(r)
    for a in (4, 1, 0, 8):
        s = ttt.step(s, a, r)
    assert ttt.position_value(s) == 1 and ttt.current_player(s) == 0
    return s


def _checker(ttt):
    calls = {"n": 0}

    def many(pairs):
        out = []
        for s, a in pairs:
            calls["n"] += 1
            child = ttt.step(s, a, random.Random(0))
            out.append(int(round(ttt.returns(child)[ttt.current_player(s)])) if ttt.is_terminal(child)
                       else -ttt.position_value(child))
        return out
    many.calls = calls
    return many


def test_checking_the_chosen_move_first_gives_the_same_walk_and_labels_a_winning_move_with_itself(ttt):
    root = _won_root(ttt)
    full = expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=None, known={})
    values = _move_values(ttt)
    check = _checker(ttt)
    fast = expand_round(ttt, root, 0, _optimal(ttt), values, depth=None, known={}, check_many=check)
    assert fast["failures"] == full["failures"] == [] and fast["labelled"].keys() == full["labelled"].keys()
    chosen = dict(zip(fast["labelled"], _optimal(ttt)([s for s, _b, _v in fast["labelled"].values()])))
    for key, (_s, best, value) in fast["labelled"].items():
        assert best == [chosen[key]] and value == 1
    assert values.calls["n"] == 0 and check.calls["n"] == len(fast["labelled"])


def test_a_checked_move_that_does_not_win_gets_the_full_label_and_is_a_failure(ttt):
    from harness.coverage import optimal_actions

    root = _won_root(ttt)
    bad = next(a for a in ttt.legal_actions(root) if a not in optimal_actions(ttt, root))
    best = _optimal(ttt)
    root_key = ttt.state_key(root)

    def chooser(states):
        return [bad if ttt.state_key(s) == root_key else best([s])[0] for s in states]
    values = _move_values(ttt)
    r = expand_round(ttt, root, 0, chooser, values, depth=None, known={}, check_many=_checker(ttt))
    full = expand_round(ttt, root, 0, chooser, _move_values(ttt), depth=None, known={})
    assert r["failures"] == full["failures"] == [root_key] and values.calls["n"] == 1
    assert sorted(r["labelled"][root_key][1]) == sorted(optimal_actions(ttt, root))


def test_a_known_winning_move_is_not_checked_again_and_a_new_choice_is_checked_and_added(ttt):
    from harness.coverage import optimal_actions

    root = _won_root(ttt)
    known: dict = {}
    check = _checker(ttt)
    expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=1, known=known, check_many=check)
    assert check.calls["n"] == 1
    expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=1, known=known, check_many=check)
    assert check.calls["n"] == 1
    winners = sorted(optimal_actions(ttt, root))
    assert len(winners) > 1
    other = winners[-1] if winners[-1] != min(winners) else winners[0]
    r = expand_round(ttt, root, 0, lambda states: [other for _ in states], _move_values(ttt), depth=1, known=known,
                     check_many=check)
    assert check.calls["n"] == 2 and r["failures"] == []
    assert sorted(known[ttt.state_key(root)]) == sorted({min(winners), other})


def test_a_drawing_move_is_never_taken_as_a_complete_label(ttt):
    from harness.coverage import optimal_actions

    root = ttt.initial_state(random.Random(0))
    values = _move_values(ttt)
    r = expand_round(ttt, root, 0, _optimal(ttt), values, depth=3, known={}, check_many=_checker(ttt))
    partial = 0
    for _key, (s, best, _v) in r["labelled"].items():
        if ttt.position_value(s) < 1:
            assert sorted(best) == sorted(optimal_actions(ttt, s))
        else:
            assert set(best) <= set(optimal_actions(ttt, s))
            partial += set(best) != set(optimal_actions(ttt, s))
    assert any(ttt.position_value(s) < 1 for s, _b, _v in r["labelled"].values()) and partial > 0


def test_growing_a_strategy_check_first_certifies_like_full_labels(ttt):
    from harness.frontier import DEFAULT_RECIPE

    root = _won_root(ttt)
    recipe = {**DEFAULT_RECIPE, "max_epochs": 1500}
    check = _checker(ttt)
    r = fit_strategy(ttt, {"mlp_hidden": [32]}, root, 0, None, 1, recipe, 6, _move_values(ttt), known={},
                     check_many=check)
    assert r["certified"] and check.calls["n"] > 0


def test_a_partially_labelled_position_gets_its_full_label_once_the_net_plays_a_losing_move_there(ttt):
    from harness.coverage import optimal_actions

    root = _won_root(ttt)
    known: dict = {}
    check = _checker(ttt)
    expand_round(ttt, root, 0, _optimal(ttt), _move_values(ttt), depth=1, known=known, check_many=check)
    assert len(known[ttt.state_key(root)]) == 1
    bad = next(a for a in ttt.legal_actions(root) if a not in optimal_actions(ttt, root))
    r = expand_round(ttt, root, 0, lambda states: [bad for _ in states], _move_values(ttt), depth=1, known=known,
                     check_many=check)
    key = ttt.state_key(root)
    assert r["failures"] == [key] and sorted(r["labelled"][key][1]) == sorted(optimal_actions(ttt, root))


def test_a_round_that_labels_nothing_new_after_a_failed_fit_stops_the_growth_as_stalled(ttt, monkeypatch):
    import harness.frontier as frontier
    import harness.strategy_fit as sf
    from harness.frontier import DEFAULT_RECIPE

    walks = iter([{"labelled": {"a": 1, "b": 2}, "failures": ["a"], "failures_by_depth": {0: 1}, "complete": True,
                   "nodes": {}},
                  {"labelled": {"a": 1}, "failures": ["a"], "failures_by_depth": {0: 1}, "complete": True,
                   "nodes": {}}])
    monkeypatch.setattr(sf, "expand_round", lambda *a, **k: next(walks))
    monkeypatch.setattr(sf, "strategy_target", lambda game, data: None)
    monkeypatch.setattr(frontier, "fit", lambda *a, **k: {"net": sf_net(ttt), "solved": False, "solved_at_epoch": None,
                                                          "best_failures": 1, "epochs_run": 1})
    r = fit_strategy(ttt, {"mlp_hidden": [4]}, ttt.initial_state(random.Random(0)), 0, None, 1, DEFAULT_RECIPE, 9,
                     _move_values(ttt), known={})
    assert not r["certified"] and r["stalled"] and len(r["rounds"]) == 2


def test_a_round_that_labels_nothing_new_after_a_SOLVED_fit_is_not_a_stall(ttt, monkeypatch):
    import harness.frontier as frontier
    import harness.strategy_fit as sf
    from harness.frontier import DEFAULT_RECIPE

    monkeypatch.setattr(sf, "expand_round", lambda *a, **k: {"labelled": {"a": 1}, "failures": ["a"],
                                                             "failures_by_depth": {0: 1}, "complete": True,
                                                             "nodes": {}})
    monkeypatch.setattr(sf, "strategy_target", lambda game, data: None)
    monkeypatch.setattr(frontier, "fit", lambda *a, **k: {"net": sf_net(ttt), "solved": True, "solved_at_epoch": 1,
                                                          "best_failures": 0, "epochs_run": 1})
    r = fit_strategy(ttt, {"mlp_hidden": [4]}, ttt.initial_state(random.Random(0)), 0, None, 1, DEFAULT_RECIPE, 4,
                     _move_values(ttt), known={})
    assert not r["certified"] and not r["stalled"] and len(r["rounds"]) == 4
