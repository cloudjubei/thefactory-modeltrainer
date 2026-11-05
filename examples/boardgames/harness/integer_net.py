"""§3.6 N2 — a quantized learned default (harness.learned_default, N1) scored in exact integers. Every tensor of a
net quantized to `bits` holds integer steps times one float32 scale; read back that way, a scale is an exact fraction
with a power-of-two denominator, so the net's outputs times one common denominator are integers — computed alike by
the rule in Python (harness.steady_state.Facts, order "net") and by the C walk (harness/c4leafwalk.c), whatever the
order of the sums. Torch-free, so the strategy builds need not load it; `features` is N1's (tests/test_integer_net.py
holds the two equal)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

KEYS = ("layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias")
LIMIT = 2 ** 63


def features(game, facts, state) -> list:
    """The cells as the mover sees them, three a cell, then three facts an action (zero for an unsafe one)."""
    own = game.current_player(state) + 1
    out = []
    for v in state.board:
        out += [int(v == own), int(v not in (0, own)), int(v == 0)]
    safe = set(facts.safe_moves(state))
    for a in range(game.num_actions):
        if a in safe:
            out += [1, int(facts._wins_in_two(state, a)), int(facts._loses_in_two(state, a))]
        else:
            out += [0, 0, 0]
    return out


def _tensor(values, bits: int) -> tuple:
    stored = np.asarray(values, dtype=np.float32)
    steps = 2 ** (bits - 1) - 1
    scale = np.float32(np.abs(stored).max()) / np.float32(steps)
    q = np.zeros(stored.shape, dtype=np.int64) if scale == 0 else np.round(stored / scale).astype(np.int64)
    if np.abs(q).max() > steps or not np.array_equal(q.astype(np.float32) * scale, stored):
        raise ValueError(f"not a {bits}-bit quantized net: a tensor is not integer steps times one float32 scale")
    return q, float(scale)


class IntegerNet:
    """A one-hidden-layer net (N1's shape: `weights` its state dict, as harness.learned_default saves it) whose
    `scores` are its outputs divided by `unit`, exactly: with h the hidden layer times D1 (the larger of its two
    scales' denominators), the output times D1 * D2 (D2 likewise for the output's scales) is
    c2 * sum(q2 * max(h, 0)) + cb2 * qb2, h = c1 * (q1 . x) + cb1 * qb1 — every constant within 64 bits, the score
    within 128."""

    def __init__(self, weights: dict, bits: int):
        self.tensors = [_tensor(weights[k], bits) for k in KEYS]
        (q1, s1), (qb1, sb1), (q2, s2), (qb2, sb2) = self.tensors
        f1, fb1, f2, fb2 = (Fraction(s) for s in (s1, sb1, s2, sb2))
        d1 = max(f1.denominator, fb1.denominator)
        d2 = max(f2.denominator, fb2.denominator)
        self.c1, self.cb1, self.c2, self.cb2 = (int(f1 * d1), int(fb1 * d1), int(f2 * d2), int(fb2 * d2 * d1))
        self.unit = Fraction(1, d1 * d2)
        self.q1, self.qb1, self.q2, self.qb2 = q1, qb1, q2, qb2
        self.hidden, self.n_in = q1.shape
        self.n_out = q2.shape[0]
        steps = 2 ** (bits - 1) - 1
        top_h = (abs(self.c1) * steps * self.n_in + abs(self.cb1) * steps)
        top_sum = steps * self.hidden * top_h
        if (max(abs(self.c1), abs(self.cb1), abs(self.c2), abs(self.cb2), top_h, top_sum) >= LIMIT
                or abs(self.c2) * top_sum + abs(self.cb2) * steps >= LIMIT * LIMIT):
            raise ValueError("the net's scales are too far apart to score in 64 bits")

    def scores(self, x) -> list:
        h = self.c1 * (self.q1 @ np.asarray(x, dtype=np.int64)) + self.cb1 * self.qb1
        s = self.q2 @ np.maximum(h, 0)
        return [self.c2 * int(sj) + self.cb2 * int(bj) for sj, bj in zip(s, self.qb2)]

    def best(self, x, moves) -> int:
        """Of `moves`, the one scored highest, the lowest on a tie."""
        s = self.scores(x)
        return min(moves, key=lambda a: (-s[a], a))
