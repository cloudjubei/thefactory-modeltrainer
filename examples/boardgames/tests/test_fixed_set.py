"""Direct tests for harness/fixed_set.py — scoring nets on ONE fixed set of exactly valued positions (the same
positions for every net, so arms are compared on equal ground), and the arithmetic that sizes a paired pilot from how
much nets of the same process vary."""
from __future__ import annotations

import random

import pytest

from games.connect4 import C4State, Connect4
from harness.fixed_set import calibration, fixed_positions, pairs_needed, score

C4 = Connect4()


def _row(moves, values):
    s = C4.initial_state(random.Random(0))
    for m in moves:
        s = C4.step(s, m)
    return {"board": list(s.board), "to_move": s.to_move, "values": {str(a): v for a, v in values.items()}}


ALL7 = {a: 0 for a in range(7)}


def test_only_full_positions_of_the_registered_side_and_plies_with_a_choice_are_kept():
    rows = [_row([], {**ALL7, 3: 1}), _row([3, 3], {**ALL7, 2: 1}), _row([3], {**ALL7, 3: 1}),
            _row([3, 3], {0: 1, 1: 0}), _row([3, 3, 2, 2], ALL7), _row([3, 3, 2, 2, 4, 4], {**ALL7, 1: 1})]
    cases = fixed_positions(rows, C4, player=0, plies=[0, 2, 4])
    assert [(ply, best) for _s, ply, values in cases for best in [max(values, key=values.get)]] == [(0, 3), (2, 2)]
    assert all(isinstance(s, C4State) for s, _p, _v in cases)


def test_a_net_s_score_counts_optimal_raw_moves_overall_and_by_ply():
    cases = [(None, 0, {0: 1, 1: 0}), (None, 2, {0: 0, 1: 1}), (None, 2, {0: 1, 1: 1, 2: 0})]
    assert score([0, 0, 1], cases) == {"positions": 3, "optimal": 2,
                                       "by_ply": {"0": {"positions": 1, "optimal": 1},
                                                  "2": {"positions": 2, "optimal": 1}}}
    with pytest.raises(ValueError, match="one move per position"):
        score([0], cases)


@pytest.mark.parametrize("sd,effect,pairs", [(0.1, 0.1, 7), (0.1, 0.05, 25), (0.02, 0.05, 1), (0.15, 0.1, 14)])
def test_the_pairs_needed_for_a_one_sided_test_at_5pct_with_80pct_power(sd, effect, pairs):
    assert pairs_needed(sd, effect) == pairs


def test_pairs_needed_refuses_a_non_positive_effect():
    with pytest.raises(ValueError, match="effect"):
        pairs_needed(0.1, 0.0)


def _net(run, arm, seed, optimal, positions=100):
    return {"run": run, "arm": arm, "seed": seed, "positions": positions, "optimal": optimal, "by_ply": {}}


def test_calibration_reads_each_arm_s_spread_and_the_seed_matched_differences_to_base():
    nets = ([_net("A", "base", s, o) for s, o in zip((1, 2, 3), (80, 70, 90))]
            + [_net("A", "x", s, o) for s, o in zip((1, 2, 3), (82, 75, 88))]
            + [_net("B", "base", s, o) for s, o in zip((7, 8), (60, 64))])
    c = calibration(nets)
    assert c["arms"]["A/base"] == {"shares": [0.8, 0.7, 0.9], "mean": 0.8, "sd": pytest.approx(0.1)}
    assert c["arms"]["B/base"]["mean"] == pytest.approx(0.62)
    assert c["paired"]["A/x"]["differences"] == pytest.approx([0.02, 0.05, -0.02])
    assert c["paired"]["A/x"]["sd"] == pytest.approx(0.035118845)
    assert c["within_arm_sd"] == pytest.approx(((0.1 ** 2 * 2 + 0.0650641 ** 2 * 2 + 0.0282843 ** 2) / 5) ** 0.5,
                                               rel=1e-4)
    assert "B/base" not in c["paired"]


def test_the_side_to_move_decides_which_positions_are_kept():
    rows = [_row([3], {**ALL7, 3: 1})]
    assert len(fixed_positions(rows, C4, player=1, plies=[1])) == 1
    assert fixed_positions(rows, C4, player=0, plies=[1]) == []
