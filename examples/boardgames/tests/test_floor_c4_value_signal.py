"""Direct tests for harness/floor_c4_value_signal.py — T14: a paired Connect-4 pilot of value-signal fixes. Per seed,
each arm's final net is read at the White positions of its own first-player tree through ply 4; each treatment must
play the optimal move at a higher share than base on enough seed pairs, by enough pooled."""
from __future__ import annotations

import pytest

from harness.floor_c4_value_signal import SPEC, pilot_report

TEST_SPEC = {"arms": {"base": {"a": 1}, "tree_value": {"a": 1, "t": True}, "n_step": {"a": 1, "n": 8}},
             "treatments": ("tree_value", "n_step"), "seeds": (1, 2, 3, 4), "era": "e" * 12,
             "measurement_fp": "m" * 12, "readout": {"plies": [0, 2, 4], "sims": 4}, "support_wins": 3,
             "refute_wins": 1, "min_gain": 0.1}
VALUES = {0: 1, 1: 0}


def _pos(arm, seed, right, ply=0, label=0, q_hat=None):
    return {"arm": arm, "seed": seed, "ply": ply, "raw": 0 if right else 1, "label": label,
            "q_hat": {str(a): q for a, q in (q_hat or {0: 0.5, 1: 0.1}).items()},
            "values": {str(a): v for a, v in VALUES.items()}}


def _readout(rights: dict):
    positions = []
    for (arm, seed), (right, n) in rights.items():
        positions += [_pos(arm, seed, i < right) for i in range(n)]
    return {"measurement_fingerprint": TEST_SPEC["measurement_fp"], "readout": TEST_SPEC["readout"],
            "eras": {arm: TEST_SPEC["era"] for arm in TEST_SPEC["arms"]},
            "configs": {arm: {**cfg, "seeds": list(TEST_SPEC["seeds"])} for arm, cfg in TEST_SPEC["arms"].items()},
            "positions": positions}


def _arms(base: list, tree: list, n_step=None, n=10):
    out = {}
    for i, seed in enumerate(TEST_SPEC["seeds"]):
        out[("base", seed)] = (base[i], n)
        out[("tree_value", seed)] = (tree[i], n)
        out[("n_step", seed)] = ((n_step or base)[i], n)
    return out


@pytest.mark.parametrize("base,tree,verdict", [
    ([5, 5, 5, 5], [7, 7, 7, 5], "supported"),
    ([5, 5, 5, 5], [7, 7, 6, 4], "supported"),
    ([5, 5, 5, 5], [6, 6, 6, 5], "inconclusive"),
    ([5, 5, 5, 5], [9, 9, 4, 4], "inconclusive"),
    ([5, 5, 5, 5], [9, 5, 5, 5], "refuted"),
    ([5, 5, 5, 5], [6, 6, 6, 1], "refuted"),
    ([5, 5, 5, 5], [5, 5, 5, 5], "refuted"),
])
def test_a_treatment_needs_wins_on_enough_seed_pairs_and_a_pooled_gain(base, tree, verdict):
    r = pilot_report(_readout(_arms(base, tree)), TEST_SPEC)
    assert r["integrity"] == [] and r["treatments"]["tree_value"]["verdict"] == verdict


def test_each_treatment_is_judged_against_base_on_its_own():
    r = pilot_report(_readout(_arms([5, 5, 5, 5], [7, 7, 7, 5], n_step=[4, 4, 4, 4])), TEST_SPEC)
    assert r["treatments"]["tree_value"]["verdict"] == "supported"
    assert r["treatments"]["n_step"] == {"verdict": "refuted", "wins": 0, "pooled": {"base": 0.5, "n_step": 0.4},
                                         "gain": -0.1}


def test_the_report_carries_each_net_s_share_the_wins_and_the_pooled_gain():
    r = pilot_report(_readout(_arms([5, 5, 5, 5], [7, 7, 7, 5])), TEST_SPEC)
    tree = r["treatments"]["tree_value"]
    assert tree["wins"] == 3 and tree["pooled"] == {"base": 0.5, "tree_value": 0.65}
    assert r["nets"]["tree_value"]["1"] == {"positions": 10, "optimal": 7}
    assert tree["gain"] == pytest.approx(0.15)


def test_shares_are_per_net_so_a_bigger_tree_does_not_buy_a_win():
    rights = _arms([5, 5, 5, 5], [7, 7, 7, 5])
    rights[("base", 1)] = (50, 100)
    r = pilot_report(_readout(rights), TEST_SPEC)
    assert r["nets"]["base"]["1"] == {"positions": 100, "optimal": 50} and r["treatments"]["tree_value"]["wins"] == 3


def test_the_value_head_and_label_readings_at_errors_are_reported_per_arm():
    e = _readout(_arms([5, 5, 5, 5], [7, 7, 7, 5]))
    e["positions"].append(_pos("base", 1, False, q_hat={0: 0.1, 1: 0.5}, label=1))
    errors = pilot_report(e, TEST_SPEC)["at_errors"]["base"]
    assert errors["errors"] == 21 and errors["value_backs"] == 1 and errors["label_optimal"] == 20


def test_the_readings_are_broken_out_by_ply():
    e = _readout(_arms([5, 5, 5, 5], [7, 7, 7, 5]))
    e["positions"].append(_pos("base", 2, True, ply=4))
    assert pilot_report(e, TEST_SPEC)["by_ply"]["base"]["4"] == {"positions": 1, "optimal": 1}


@pytest.mark.parametrize("breakage", ["fingerprint", "era", "readout", "configs", "missing net", "extra seed", "ply",
                                      "arm"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run_for_every_treatment(breakage):
    e = _readout(_arms([5, 5, 5, 5], [7, 7, 7, 5]))
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "era":
        e["eras"]["base"] = "0" * 12
    elif breakage == "readout":
        e["readout"] = {"plies": [0, 2], "sims": 4}
    elif breakage == "configs":
        e["configs"]["n_step"] = {"a": 1, "seeds": [1, 2, 3, 4]}
    elif breakage == "missing net":
        e["positions"] = [p for p in e["positions"] if not (p["arm"] == "base" and p["seed"] == 4)]
    elif breakage == "extra seed":
        e["positions"].append(_pos("base", 9, True))
    elif breakage == "ply":
        e["positions"][0]["ply"] = 6
    else:
        e["positions"][0]["arm"] = "other"
    r = pilot_report(e, TEST_SPEC)
    assert r["integrity"]
    assert r["treatments"] == {"tree_value": {"verdict": "not_run"}, "n_step": {"verdict": "not_run"}}


def test_the_registered_pilot_is_t10_s_process_at_depth_6_for_20_iterations_with_two_value_signal_fixes():
    from harness.floor_c4 import CONFIG

    base = {**CONFIG, "iterations": 20, "strategy_tree": {"player": 0, "depth": 6}, "relabel_workers": 4,
            "stop_on_agreement": False}
    assert SPEC["arms"] == {"base": base, "tree_value": {**base, "tree_value_target": True},
                            "n_step": {**base, "value_n_step": 8, "target_refresh": 2}}
    assert SPEC["treatments"] == ("tree_value", "n_step")
    assert SPEC["seeds"] == (401, 402, 403, 404) and SPEC["readout"] == {"plies": [0, 2, 4], "sims": 200}
    assert (SPEC["support_wins"], SPEC["refute_wins"], SPEC["min_gain"]) == (3, 1, 0.1)
