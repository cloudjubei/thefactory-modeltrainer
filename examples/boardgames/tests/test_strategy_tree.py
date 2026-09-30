"""Direct tests for harness/strategy_tree.py — the solver-free pieces of §C.49 task 6: the positions a net's own RAW
strategy reaches against every reply (the coverage the process trains on), and the solver-free reading of whether
the net's raw move agrees with its own search there."""
from __future__ import annotations

import random

import pytest

from harness.strategy_tree import disagreements, strategy_tree_positions


@pytest.fixture(scope="module")
def ttt():
    from games.tictactoe import TicTacToe

    return TicTacToe()


def _first_legal(game):
    return lambda states: [game.legal_actions(s)[0] for s in states]


def _reference(game, root, player, choose, depth):
    seen, out, level, ply = set(), [], [root], 0
    while level and (depth is None or ply < depth):
        nxt = []
        for s in level:
            k = game.state_key(s)
            if k in seen or game.is_terminal(s):
                continue
            seen.add(k)
            if game.current_player(s) == player:
                out.append(k)
                moves = choose([s])
            else:
                moves = game.legal_actions(s)
            nxt.extend(game.step(s, a, random.Random(0)) for a in moves)
        level, ply = nxt, ply + 1
    return out


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("depth", [None, 1, 2, 3, 5])
def test_the_walk_returns_every_position_of_the_player_s_own_tree_once_in_breadth_first_order(ttt, player, depth):
    root = ttt.initial_state(random.Random(0))
    got = strategy_tree_positions(ttt, root, player, _first_legal(ttt), depth)
    assert [ttt.state_key(s) for s in got] == _reference(ttt, root, player, _first_legal(ttt), depth)
    assert all(ttt.current_player(s) == player and not ttt.is_terminal(s) for s in got)


def test_the_chooser_is_asked_once_per_ply_with_that_ply_s_positions(ttt):
    root = ttt.initial_state(random.Random(0))
    batches = []
    first = _first_legal(ttt)

    def choose(states):
        batches.append(len(states))
        return first(states)
    got = strategy_tree_positions(ttt, root, 0, choose, None)
    assert sum(batches) == len(got) and len(batches) < len(got)


def test_a_depth_of_zero_is_refused(ttt):
    with pytest.raises(ValueError, match="depth"):
        strategy_tree_positions(ttt, ttt.initial_state(random.Random(0)), 0, _first_legal(ttt), 0)


@pytest.mark.parametrize("move,label,disagrees", [
    (2, [0.0, 0.0, 1.0], False),
    (1, [0.0, 0.4, 0.6], False),
    (1, [0.0, 0.29, 0.61], True),
    (0, [0.1, 0.45, 0.45], True),
    (1, [0.1, 0.45, 0.45], False),
    (0, [0.5, 0.25, 0.25], False),
    (1, [0.25, 0.25, 0.5], False),
])
def test_a_raw_move_disagrees_only_when_its_search_share_is_under_half_the_top_move_s(move, label, disagrees):
    assert disagreements([move], [label]) == int(disagrees)


def test_disagreements_are_counted_over_every_position():
    moves = [0, 1, 2, 0]
    labels = [[1, 0, 0], [0.9, 0.1, 0], [0.2, 0.3, 0.5], [0.3, 0.3, 0.4]]
    assert disagreements(moves, labels) == 1
    with pytest.raises(ValueError, match="same length"):
        disagreements([0], [])


class _Fixed:
    def __init__(self, logits):
        import torch

        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.training = True
        self.modes = []

    def eval(self):
        self.modes.append("eval")
        self.training = False

    def train(self, mode=True):
        self.modes.append(mode)
        self.training = mode

    def __call__(self, x):
        return self.logits.expand(len(x), -1), None


def test_the_raw_chooser_is_the_argmax_over_legal_moves_and_leaves_the_net_as_it_found_it(ttt):
    from harness.strategy_tree import raw_chooser

    r = random.Random(0)
    root = ttt.initial_state(r)
    taken = ttt.step(root, 4, r)
    net = _Fixed([0, 1, 2, 3, 9, 5, 6, 7, 8])
    assert raw_chooser(ttt, net)([root, taken]) == [4, 8]
    assert net.modes == ["eval", True] and net.training
    net.eval()
    net.modes = []
    raw_chooser(ttt, net)([root])
    assert net.modes == ["eval", False]
