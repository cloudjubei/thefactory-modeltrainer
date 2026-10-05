"""Direct tests for harness/depth_cost.py — net cost against table cost as the certified depth grows. A complete table
is a list of moves read in the strategy tree's canonical walk order (no index); a hybrid stores its net plus an
exception wherever the net's move is not optimal, as the cheaper of a sparse index or a one-bit-per-position mask. The
judge reads the net's exception share at an early and a late ply of its own tree, and the break-even precision (bits
per weight at which the hybrid costs the same as the table) at the judged depth."""
from __future__ import annotations

import pytest

from harness.depth_cost import SPEC, depth_report, exception_bits, through, walk_table_bits

TEST_SPEC = {"era": "e" * 12, "measurement_fp": "m" * 12, "seeds": [1, 2, 3, 4], "deep": [1, 2],
             "depths": [3, 5], "early_ply": 2, "late_ply": 4, "drop": 0.02, "judged_depth": 5, "break_even_at": 4,
             "bits": [32, 8], "actions": 7}


@pytest.mark.parametrize("entries,actions,bits", [(10, 7, 30), (10, 8, 30), (10, 9, 40), (5, 2, 5), (5, 1, 0),
                                                  (0, 7, 0)])
def test_a_complete_table_costs_one_move_per_entry_in_walk_order(entries, actions, bits):
    assert walk_table_bits(entries, actions) == bits


@pytest.mark.parametrize("exceptions,positions,bits", [(0, 1000, 0), (1, 1000, 13), (50, 1000, 650),
                                                        (150, 1000, 1000 + 450), (500, 1000, 2500), (3, 4, 4 + 9)])
def test_exceptions_cost_the_cheaper_of_a_sparse_index_and_a_position_mask(exceptions, positions, bits):
    assert exception_bits(exceptions, positions, 7) == bits


def test_an_empty_tree_has_no_exceptions_to_pay_for():
    assert exception_bits(0, 0, 7) == 0


def test_exceptions_outside_the_tree_are_refused():
    with pytest.raises(ValueError, match="exceptions"):
        exception_bits(11, 10, 7)


def test_through_sums_the_plies_before_the_depth():
    assert through({"0": 1, "2": 4, "4": 9, "6": 30}, 5) == 14
    assert through({"0": 1, "2": 4}, 1) == 1 and through({}, 9) == 0
    assert through({"0": 1, "2": 4, "4": 9}, 4) == 5


def _net(seed, horizon, positions, exceptions, params=1000, certified=True):
    return {"seed": seed, "horizon": horizon, "params": params, "certified": certified,
            "positions_by_ply": {str(p): v for p, v in positions.items()},
            "exceptions_by_ply": {str(p): v for p, v in exceptions.items()}}


TABLE = {"0": 1, "2": 6, "4": 36}


def _readout(late=(1, 1), early=(5, 5), table=TABLE, deep_positions=36):
    nets = [_net(s, 5, {0: 1, 2: 6, 4: deep_positions}, {0: 0, 2: e, 4: l})
            for s, e, l in zip((1, 2), early, late)]
    nets += [_net(s, 3, {0: 1, 2: 6}, {0: 1, 2: 2}) for s in (3, 4)]
    return {"measurement_fingerprint": "m" * 12, "run": {"training_fingerprint": "e" * 12},
            "table": {"horizon": 5, "by_ply": dict(table)}, "nets": nets}


def test_the_late_ply_share_falling_by_the_registered_drop_supports_the_trend():
    r = depth_report(_readout(), TEST_SPEC)
    assert r["integrity"] == [] and r["trend"]["verdict"] == "supported"
    assert r["trend"]["early"] == pytest.approx(10 / 12) and r["trend"]["late"] == pytest.approx(2 / 72)


def test_a_late_share_no_lower_refutes_the_trend_and_a_small_drop_is_inconclusive():
    flat = _readout(late=(30, 30), early=(5, 5))
    assert depth_report(flat, TEST_SPEC)["trend"]["verdict"] == "refuted"
    small = _readout(late=(30, 29), early=(5, 5))
    r = depth_report(small, TEST_SPEC)["trend"]
    assert r["late"] == pytest.approx(59 / 72) and r["verdict"] == "inconclusive"


def test_break_even_is_the_bits_per_weight_at_which_net_and_exceptions_cost_the_table():
    r = depth_report(_readout(table={"0": 1, "2": 6, "4": 2000}), TEST_SPEC)
    deep = [row for row in r["rows"] if row["seed"] == 1 and row["depth"] == 5][0]
    assert deep["table_bits"] == 2007 * 3 and deep["exception_bits"] == 6 * (6 + 3)
    assert deep["break_even"] == pytest.approx((2007 * 3 - 54) / 1000)
    assert deep["hybrid_bits"] == {"32": 32000 + 54, "8": 8000 + 54}


def test_every_deep_net_must_break_even_above_the_registered_precision_for_support():
    big = {"0": 1, "2": 6, "4": 2000}
    assert depth_report(_readout(table=big), TEST_SPEC)["break_even"]["verdict"] == "supported"
    assert depth_report(_readout(), TEST_SPEC)["break_even"]["verdict"] == "refuted"
    split = _readout(table=big)
    split["nets"][1]["params"] = 100000
    r = depth_report(split, TEST_SPEC)["break_even"]
    assert r["verdict"] == "inconclusive" and len(r["values"]) == 2


def test_rows_cover_every_depth_each_net_reaches():
    r = depth_report(_readout(), TEST_SPEC)
    assert sorted((row["seed"], row["depth"]) for row in r["rows"]) == [(1, 3), (1, 5), (2, 3), (2, 5), (3, 3), (4, 3)]
    shallow = [row for row in r["rows"] if row["seed"] == 3][0]
    assert shallow["positions"] == 7 and shallow["exceptions"] == 3 and shallow["table_entries"] == 7


@pytest.mark.parametrize("breakage", ["fingerprint", "era", "missing seed", "shallow deep net", "deep table short",
                                      "uncertified", "deep net too shallow for the late ply"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _readout()
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "era":
        e["run"]["training_fingerprint"] = "0" * 12
    elif breakage == "missing seed":
        e["nets"] = e["nets"][:-1]
    elif breakage == "shallow deep net":
        e["nets"][0]["horizon"] = 3
    elif breakage == "deep table short":
        e["table"]["horizon"] = 3
    elif breakage == "uncertified":
        e["nets"][2]["certified"] = False
    else:
        del e["nets"][0]["positions_by_ply"]["4"]
    r = depth_report(e, TEST_SPEC)
    assert r["integrity"] and r["trend"] == {"verdict": "not_run"} and r["break_even"] == {"verdict": "not_run"}


def test_the_registered_study_reads_h3_s_nets_three_of_them_to_depth_13():
    assert SPEC["seeds"] == [481, 482, 483, 484, 485, 486, 487] and SPEC["deep"] == [481, 482, 483]
    assert SPEC["depths"] == [9, 11, 13] and (SPEC["early_ply"], SPEC["late_ply"]) == (8, 12)
    assert (SPEC["drop"], SPEC["judged_depth"], SPEC["break_even_at"]) == (0.02, 13, 4)
    assert SPEC["era"] == "33939d5e2d76" and SPEC["run"] == "c49_H3_deep" and SPEC["actions"] == 7
