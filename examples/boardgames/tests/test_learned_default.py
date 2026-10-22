"""Direct tests for harness/learned_default.py — a small net that picks the default move among the safe ones from the
board (as the mover sees it) and each move's tactical facts, trained to put its weight on winning moves, its weights
quantized and counted in bits."""
from __future__ import annotations

import pytest
import torch

from games.tictactoe import TicTacToe, TTTState
from harness.learned_default import (Net, features, n_features, net_bits, picks, quantized, train)
from harness.steady_state import Facts

GAME = TicTacToe()


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


def test_the_board_is_read_as_the_mover_sees_it_then_each_move_s_facts():
    facts = Facts(GAME)
    x = features(GAME, facts, _ttt((0, 1), (2, 3)))
    o = features(GAME, facts, _ttt((0, 1, 5), (2, 3)))
    assert len(x) == len(o) == n_features(GAME) == 9 * 3 + 9 * 3
    assert x[0:3] == [1, 0, 0] and x[6:9] == [0, 1, 0] and x[12:15] == [0, 0, 1]
    assert o[0:3] == [0, 1, 0] and o[6:9] == [1, 0, 0]
    safe = facts.safe_moves(_ttt((0, 1), (2, 3)))
    assert [x[27 + 3 * a] for a in range(9)] == [1 if a in safe else 0 for a in range(9)]


def test_a_move_that_wins_in_two_and_one_that_loses_in_two_are_marked():
    facts = Facts(GAME)
    x = features(GAME, facts, _ttt((0, 1), (2, 3)))
    assert [a for a in range(9) if x[27 + 3 * a + 1]] == [4]
    assert [a for a in range(9) if x[27 + 3 * a + 2]] == [6, 7, 8]
    assert [x[27 + 3 * a + 1] + x[27 + 3 * a + 2] for a in (0, 1, 2, 3)] == [0, 0, 0, 0]


def test_the_net_s_bits_are_its_weights_at_their_width_and_a_scale_a_tensor():
    net = Net(10, 4, 3)
    assert net_bits(net, 4) == (10 * 4 + 4 + 4 * 3 + 3) * 4 + 4 * 32
    assert net_bits(net, 8) == (10 * 4 + 4 + 4 * 3 + 3) * 8 + 4 * 32


def test_quantized_weights_take_at_most_their_levels_and_keep_the_largest():
    torch.manual_seed(0)
    net = Net(10, 4, 3)
    q = quantized(net, 4)
    for before, after in zip(net.parameters(), q.parameters()):
        scale = before.detach().abs().max() / 7
        steps = (after.detach() / scale).round()
        assert torch.allclose(after.detach(), steps * scale)
        assert steps.abs().max() == 7 and len(steps.unique()) <= 15
    assert q is not net


def test_picks_take_the_best_safe_move_and_never_an_unsafe_one():
    net = Net(2, 2, 3)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.layers[2].bias.copy_(torch.tensor([5.0, 1.0, 3.0]))
    x = torch.zeros(2, 2)
    safe = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.bool)
    assert picks(net, x, safe).tolist() == [0, 2]


def test_a_tie_goes_to_the_lowest_safe_move():
    net = Net(2, 2, 3)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.layers[2].bias.copy_(torch.tensor([3.0, 3.0, 3.0]))
    safe = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.bool)
    assert picks(net, torch.zeros(2, 2), safe).tolist() == [0, 1]


def test_the_hidden_layer_learns_what_no_linear_score_can():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).repeat(25, 1)
    safe = torch.ones(100, 2, dtype=torch.bool)
    won = torch.zeros(100, 2, dtype=torch.bool)
    for i in range(100):
        won[i, int(x[i, 0] != x[i, 1])] = True
    net = train(x, safe, won, hidden=8, epochs=200, batch=20, lr=0.05, seed=0)
    assert torch.equal(picks(net, x, safe), won.int().argmax(dim=1))


def test_training_learns_to_put_its_weight_on_winning_moves():
    torch.manual_seed(1)
    x = torch.eye(3).repeat(67, 1)[:200]
    safe = torch.ones(200, 3, dtype=torch.bool)
    won = torch.zeros(200, 3, dtype=torch.bool)
    for i in range(200):
        won[i, i % 3] = True
    net = train(x, safe, won, hidden=8, epochs=60, batch=32, lr=0.05, seed=0)
    assert picks(net, x, safe).tolist() == [i % 3 for i in range(200)]


def test_training_is_reproducible_from_its_seed():
    x = torch.rand(40, 5)
    safe = torch.ones(40, 3, dtype=torch.bool)
    won = torch.zeros(40, 3, dtype=torch.bool)
    won[:, 1] = True
    a = train(x, safe, won, hidden=4, epochs=3, batch=8, lr=0.01, seed=7)
    b = train(x, safe, won, hidden=4, epochs=3, batch=8, lr=0.01, seed=7)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_a_position_with_no_winning_safe_move_cannot_train():
    with pytest.raises(ValueError):
        train(torch.rand(2, 3), torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.bool), hidden=2,
              epochs=1, batch=2, lr=0.01, seed=0)
