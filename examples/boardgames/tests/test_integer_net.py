"""Direct tests for harness/integer_net.py — a quantized net (harness.learned_default) read back as integer steps and
float32 scales, scored in exact integers so the Python rule and the C walk play the same move: the steps times the
scales give back every stored weight, the scores are the exact outputs times one positive unit, ties go to the lowest
move, and its features are N1's."""
from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest
import torch

from games.connect4 import Connect4
from harness.integer_net import IntegerNet, features
from harness.learned_default import Net, features as n1_features, quantized
from harness.steady_state import Facts

GAME = Connect4()


def _weights(n_in=9, hidden=5, n_out=4, seed=0, bits=4):
    torch.manual_seed(seed)
    return {k: v.tolist() for k, v in quantized(Net(n_in, hidden, n_out), bits).state_dict().items()}


def _steps_times_scale(values, bits):
    stored = np.asarray(values, dtype=np.float32)
    scale = np.float32(np.abs(stored).max()) / np.float32(2 ** (bits - 1) - 1)
    exact = Fraction(float(scale))
    return [[int(round(float(v / scale))) * exact for v in row] for row in np.atleast_2d(stored)]


def _exact(weights, bits, x):
    """The outputs of the net whose weights are each tensor's integer steps times its float32 scale, as fractions."""
    w1, b1, w2, b2 = (_steps_times_scale(weights[k], bits)
                      for k in ("layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias"))
    b1, b2 = b1[0], b2[0]
    h = [max(sum(w * xi for w, xi in zip(row, x)) + b, 0) for row, b in zip(w1, b1)]
    return [sum(w * hk for w, hk in zip(row, h)) + b for row, b in zip(w2, b2)]


@pytest.mark.parametrize("bits,seed,shape", [(4, 0, (9, 5, 4)), (3, 1, (12, 3, 7)), (6, 2, (5, 8, 2))])
def test_the_steps_times_the_scales_give_back_every_weight(bits, seed, shape):
    weights = _weights(*shape, seed=seed, bits=bits)
    net = IntegerNet(weights, bits)
    steps = 2 ** (bits - 1) - 1
    for (q, scale), key in zip(net.tensors, ("layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias")):
        stored = np.asarray(weights[key], dtype=np.float32)
        assert np.abs(np.asarray(q)).max() <= steps
        assert np.array_equal(np.asarray(q, dtype=np.float32) * np.float32(scale), stored)
    assert (net.n_in, net.hidden, net.n_out) == shape


def test_a_net_that_is_not_quantized_is_refused():
    torch.manual_seed(0)
    weights = {k: v.tolist() for k, v in Net(9, 5, 4).state_dict().items()}
    with pytest.raises(ValueError, match="not a 4-bit quantized net"):
        IntegerNet(weights, 4)


@pytest.mark.parametrize("bits,seed,shape", [(4, 0, (9, 5, 4)), (3, 1, (12, 3, 7)), (6, 2, (5, 8, 2))])
def test_the_scores_are_the_exact_outputs_in_one_positive_unit(bits, seed, shape):
    weights = _weights(*shape, seed=seed, bits=bits)
    net = IntegerNet(weights, bits)
    rng = random.Random(seed)
    assert net.unit > 0
    for _ in range(50):
        x = [rng.randrange(2) for _ in range(shape[0])]
        assert [s * net.unit for s in net.scores(x)] == _exact(weights, bits, x)


def test_the_scores_follow_the_float_net():
    torch.manual_seed(3)
    q = quantized(Net(9, 5, 4), 4)
    net = IntegerNet({k: v.tolist() for k, v in q.state_dict().items()}, 4)
    rng = random.Random(3)
    for _ in range(50):
        x = [rng.randrange(2) for _ in range(9)]
        with torch.no_grad():
            out = q(torch.tensor([x], dtype=torch.float32))[0].tolist()
        assert [float(s * net.unit) for s in net.scores(x)] == pytest.approx(out, abs=1e-5)


def test_the_best_move_is_the_highest_score_the_lowest_on_a_tie():
    weights = {"layers.0.weight": [[0.0, 0.0]], "layers.0.bias": [0.0],
               "layers.2.weight": [[0.0]] * 4, "layers.2.bias": [1.0, 3.0, 3.0, -2.0]}
    net = IntegerNet(weights, 3)
    assert net.scores([1, 0])[1] == net.scores([1, 0])[2] > net.scores([1, 0])[0]
    assert net.best([1, 0], [0, 1, 2, 3]) == 1
    assert net.best([1, 0], [3, 2]) == 2
    assert net.best([1, 0], [0, 3]) == 0
    assert net.best([1, 0], [3]) == 3


def test_a_zero_tensor_is_zero_steps_at_scale_zero():
    net = IntegerNet({"layers.0.weight": [[0.0, 0.0]], "layers.0.bias": [0.0],
                      "layers.2.weight": [[0.0]] * 3, "layers.2.bias": [1.0, 0.0, -1.0]}, 4)
    for q, scale in net.tensors[:3]:
        assert scale == 0 and not np.asarray(q).any()
    assert net.best([1, 1], [0, 1, 2]) == 0


def test_scales_too_far_apart_for_64_bits_are_refused():
    weights = {"layers.0.weight": [[1.0, 3.0]], "layers.0.bias": [3e-30],
               "layers.2.weight": [[3.0]], "layers.2.bias": [3.0]}
    with pytest.raises(ValueError, match="64 bits"):
        IntegerNet(weights, 3)


def _positions(seed, count):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        s = GAME.initial_state()
        for _ in range(rng.randrange(0, 30)):
            if GAME.is_terminal(s):
                break
            s = GAME.step(s, rng.choice(GAME.legal_actions(s)))
        if not GAME.is_terminal(s):
            out.append(s)
    return out


@pytest.mark.parametrize("depth", [1, 2])
def test_the_features_are_n1_s(depth):
    facts = Facts(GAME, safe_depth=depth)
    positions = _positions(depth, 200)
    assert any(s.to_move == 1 for s in positions) and any(len(facts.safe_moves(s)) < 7 for s in positions)
    assert any(facts._wins_in_two(s, a) for s in positions for a in facts.safe_moves(s))
    assert any(facts._loses_in_two(s, a) for s in positions for a in facts.safe_moves(s))
    for s in positions:
        assert features(GAME, facts, s) == n1_features(GAME, facts, s)
