"""Direct tests for harness/floor_h3.py — H3: train the plies the net must carry. The H3 arm walks the strategy tree
from the table's frontier deep enough to cover plies 6 and 8; its nets are compared with H2's table-arm nets of the
same seeds at a fixed set of ply-8 positions, and its learning curve at the ply-6 positions is read for whether it is
still rising at the end of training."""
from __future__ import annotations

import pytest

from harness.floor_h3 import SPEC, h3_report

SEEDS = (1, 2, 3, 4, 5, 6, 7)
TEST_SPEC = {"arms": {"h2": {"a": 2}, "h3": {"a": 3}}, "runs": {"h2": "P/h2", "h3": "Q/h3"}, "seeds": SEEDS,
             "era": "e" * 12, "measurement_fp": "m" * 12, "s6_positions": 100, "s8_positions": 200,
             "iterations": 10, "early": [5, 6], "late": [9, 10], "rising_at": 0.015, "flat_at": 0.005, "alpha": 0.05}


def _curve(early_share, late_share):
    n = TEST_SPEC["iterations"] + 1
    shares = [early_share] * 6 + [late_share] * (n - 6)
    return [{"positions": 1000, "optimal": round(s * 1000)} for s in shares]


def _net(arm, seed, s8_optimal, s6_optimal=80, curve=None):
    n = {"arm": arm, "seed": seed, "s6": {"positions": 100, "optimal": s6_optimal},
         "s8": {"positions": 200, "optimal": s8_optimal}, "description": {"entries": 10, "bits": 1000}}
    if arm == "h3":
        n["carried_curve"] = curve if curve is not None else _curve(0.80, 0.82)
    return n


BASE = [150, 140, 160, 145, 155, 150, 148]


def _readout(h3_s8, h2_s8=BASE, curve=None):
    nets = [_net("h2", s, o) for s, o in zip(SEEDS, h2_s8)] + [_net("h3", s, o, curve=curve)
                                                             for s, o in zip(SEEDS, h3_s8)]
    return {"measurement_fingerprint": TEST_SPEC["measurement_fp"], "s6_positions": 100, "s8_positions": 200,
            "runs": {arm: {"training_fingerprint": TEST_SPEC["era"],
                           "config": {**cfg, "seeds": list(SEEDS), "certify_depth": 6}}
                     for arm, cfg in TEST_SPEC["arms"].items()},
            "nets": nets}


def test_an_h3_net_better_at_ply_8_on_every_seed_supports_the_gain():
    r = h3_report(_readout([b + 10 for b in BASE]), TEST_SPEC)
    assert r["integrity"] == [] and r["s8"]["verdict"] == "supported"
    assert r["s8"]["p"] == pytest.approx(1 / 128) and r["s8"]["mean"] == pytest.approx(0.05)


def test_no_gain_at_ply_8_is_refuted_and_a_noisy_one_is_inconclusive():
    assert h3_report(_readout(BASE), TEST_SPEC)["s8"]["verdict"] == "refuted"
    noisy = [b + d for b, d in zip(BASE, (12, -8, 10, -6, 4, -2, 2))]
    assert h3_report(_readout(noisy), TEST_SPEC)["s8"]["verdict"] == "inconclusive"


@pytest.mark.parametrize("early,late,verdict", [(0.80, 0.82, "supported"), (0.80, 0.815, "supported"),
                                                (0.80, 0.81, "inconclusive"), (0.80, 0.805, "refuted"),
                                                (0.80, 0.79, "refuted")])
def test_the_curve_reading_is_the_pooled_late_minus_early_share(early, late, verdict):
    r = h3_report(_readout([b + 10 for b in BASE], curve=_curve(early, late)), TEST_SPEC)
    assert r["curve"]["verdict"] == verdict and r["curve"]["gain"] == pytest.approx(late - early)


def test_early_and_late_are_averaged_over_their_registered_passes():
    curve = _curve(0.80, 0.80)
    curve[4] = {"positions": 1000, "optimal": 900}
    curve[9] = {"positions": 1000, "optimal": 700}
    r = h3_report(_readout([b + 10 for b in BASE], curve=curve), TEST_SPEC)
    assert r["curve"]["gain"] == pytest.approx(-0.1)


def test_the_ply_6_shares_and_the_description_sizes_are_reported_per_arm():
    r = h3_report(_readout([b + 10 for b in BASE]), TEST_SPEC)
    assert r["s6"] == {"h2": [0.8] * 7, "h3": [0.8] * 7}
    assert r["description"]["h3"] == [{"entries": 10, "bits": 1000}] * 7


@pytest.mark.parametrize("breakage", ["fingerprint", "era", "config", "s6 size", "readout s6 size", "s8 size",
                                      "missing net", "extra net", "no curve", "short curve"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _readout([b + 10 for b in BASE])
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "era":
        e["runs"]["h3"]["training_fingerprint"] = "0" * 12
    elif breakage == "config":
        e["runs"]["h2"]["config"]["a"] = 9
    elif breakage == "s6 size":
        e["nets"][0]["s6"]["positions"] = 99
    elif breakage == "readout s6 size":
        e["s6_positions"] = 99
    elif breakage == "extra net":
        e["nets"].append(_net("h2", 99, 150))
    elif breakage == "s8 size":
        e["s8_positions"] = 199
    elif breakage == "missing net":
        e["nets"] = e["nets"][1:]
    elif breakage == "no curve":
        del e["nets"][-1]["carried_curve"]
    else:
        e["nets"][-1]["carried_curve"] = e["nets"][-1]["carried_curve"][:-1]
    r = h3_report(e, TEST_SPEC)
    assert r["integrity"] and r["s8"] == {"verdict": "not_run"} and r["curve"] == {"verdict": "not_run"}


def test_the_registered_comparison_is_h3_against_h2_s_table_nets_on_the_same_seeds():
    from harness.floor_h2 import SPEC as H2_SPEC

    h2 = H2_SPEC["arms"]["table"]
    assert SPEC["arms"] == {"h2": h2, "h3": {**h2, "strategy_tree": {"player": 0, "depth": 4}}}
    assert SPEC["runs"] == {"h2": "c49_H2/table", "h3": "c49_H3/deep"} and SPEC["seeds"] == H2_SPEC["seeds"]
    assert SPEC["era"] == H2_SPEC["era"] and SPEC["s6_positions"] == 284 and SPEC["s8_positions"] is None
    assert (SPEC["early"], SPEC["late"], SPEC["rising_at"], SPEC["flat_at"]) == ([11, 12, 13, 14, 15],
                                                                                  [16, 17, 18, 19, 20], 0.015, 0.005)


def test_an_open_s8_size_still_needs_every_net_scored_on_the_same_set():
    spec = {**TEST_SPEC, "s8_positions": None}
    assert h3_report(_readout([b + 10 for b in BASE]), spec)["integrity"] == []
    e = _readout([b + 10 for b in BASE])
    e["nets"][3]["s8"]["positions"] = 150
    assert h3_report(e, spec)["s8"] == {"verdict": "not_run"}
    e = _readout([b + 10 for b in BASE])
    e["s8_positions"] = 0
    for n in e["nets"]:
        n["s8"]["positions"] = 0
    assert h3_report(e, spec)["s8"] == {"verdict": "not_run"}


def test_a_p_just_above_alpha_is_not_supported():
    r = h3_report(_readout([b + d for b, d in zip(BASE, (2, 2, 2, 2, 2, 2, -2))]), TEST_SPEC)
    assert r["s8"]["p"] == pytest.approx(8 / 128) and r["s8"]["verdict"] == "inconclusive"
