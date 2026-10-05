"""Direct tests for harness/floor_h2.py — H2: the hybrid's table in the training loop. With the exact opening table
fixing the first player's moves before the horizon, every net faces the same positions one ply past the table; the
table-trained net must play the optimal move there more often than the base net of the same seed."""
from __future__ import annotations

import pytest

from harness.floor_h2 import SPEC, h2_report

TEST_SPEC = {"arms": {"base": {"a": 0}, "table": {"a": 1}}, "seeds": (1, 2, 3, 4, 5, 6, 7), "prefix": "P",
             "era": "e" * 12, "measurement_fp": "m" * 12, "horizon": 5, "carried_positions": 100, "alpha": 0.05}


def _net(arm, seed, optimal, certified=False):
    return {"arm": arm, "seed": seed, "carried": {"positions": 100, "optimal": optimal},
            "certificate": {"depth": TEST_SPEC["horizon"] + 4, "certified": certified,
                            "failures": 0 if certified else 3, "failures_by_ply": {} if certified else {"8": 3}}}


BASE = [80, 82, 85, 79, 81, 83, 80]


def _evidence(table_optimals, base=BASE):
    nets = ([_net("base", s, o) for s, o in zip(TEST_SPEC["seeds"], base)]
            + [_net("table", s, o) for s, o in zip(TEST_SPEC["seeds"], table_optimals)])
    return {"measurement_fingerprint": TEST_SPEC["measurement_fp"], "horizon": TEST_SPEC["horizon"],
            "carried_positions": TEST_SPEC["carried_positions"],
            "runs": {f"P/{arm}": {"training_fingerprint": TEST_SPEC["era"],
                                  "config": {**cfg, "seeds": list(TEST_SPEC["seeds"]), "certify_depth": 6}}
                     for arm, cfg in TEST_SPEC["arms"].items()},
            "nets": nets}


def test_a_table_net_better_on_every_seed_is_supported_with_the_exact_p():
    r = h2_report(_evidence([b + 5 for b in BASE]), TEST_SPEC)
    assert r["integrity"] == [] and r["verdict"] == "supported"
    assert r["p"] == pytest.approx(1 / 128) and r["mean"] == pytest.approx(0.05)


def test_no_gain_is_refuted_and_a_noisy_gain_is_inconclusive():
    assert h2_report(_evidence(BASE), TEST_SPEC)["verdict"] == "refuted"
    noisy = [b + d for b, d in zip(BASE, (6, -4, 5, -3, 2, -1, 1))]
    r = h2_report(_evidence(noisy), TEST_SPEC)
    assert r["mean"] > 0 and r["verdict"] == "inconclusive"


def test_shares_and_certificates_are_reported_per_arm():
    e = _evidence([b + 5 for b in BASE])
    e["nets"][-1]["certificate"] = {"depth": 9, "certified": True, "failures": 0, "failures_by_ply": {}}
    r = h2_report(e, TEST_SPEC)
    assert r["certified"] == {"base": 0, "table": 1}
    assert r["shares"]["table"][0] == pytest.approx(0.85) and len(r["shares"]["base"]) == 7
    assert r["failures_by_ply"]["base"] == {"8": 21}


@pytest.mark.parametrize("breakage", ["fingerprint", "era", "config", "horizon", "carried set", "missing net",
                                      "extra net", "certificate depth"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _evidence([b + 5 for b in BASE])
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "era":
        e["runs"]["P/table"]["training_fingerprint"] = "0" * 12
    elif breakage == "config":
        e["runs"]["P/base"]["config"]["a"] = 5
    elif breakage == "horizon":
        e["horizon"] = 3
    elif breakage == "carried set":
        e["nets"][3]["carried"]["positions"] = 99
    elif breakage == "missing net":
        e["nets"] = e["nets"][1:]
    elif breakage == "extra net":
        e["nets"].append(_net("base", 99, 80))
    else:
        e["nets"][0]["certificate"]["depth"] = 7
    r = h2_report(e, TEST_SPEC)
    assert r["integrity"] and r["verdict"] == "not_run"


def test_the_registered_comparison_trains_base_and_table_arms_on_seven_fresh_seeds():
    from harness.floor_c4_powered import SPEC as T17_SPEC

    base = T17_SPEC["arms"]["base"]
    assert SPEC["arms"]["base"] == base
    assert SPEC["arms"]["table"] == {**base, "strategy_tree": {"player": 0, "depth": 2},
                                     "opening_table": {"horizon": 5, "entries": 56, "frontier": 44}}
    assert SPEC["seeds"] == tuple(range(481, 488)) and SPEC["prefix"] == "c49_H2"
    assert (SPEC["horizon"], SPEC["carried_positions"], SPEC["alpha"]) == (5, 284, 0.05)


def test_the_gain_is_paired_by_seed():
    base = [55, 90, 55, 90, 55, 90, 70]
    r = h2_report(_evidence([b + 5 for b in base], base=base), TEST_SPEC)
    assert r["verdict"] == "supported" and r["differences"] == pytest.approx([0.05] * 7)


def test_a_p_just_above_alpha_is_not_supported():
    r = h2_report(_evidence([b + d for b, d in zip(BASE, (1, 1, 1, 1, 1, 1, -1))]), TEST_SPEC)
    assert r["p"] == pytest.approx(8 / 128) and r["verdict"] == "inconclusive"
