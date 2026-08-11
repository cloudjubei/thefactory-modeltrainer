import random

import torch
import torch.nn.functional as F

from games.connect4 import Connect4
from harness.agents import HeuristicAgent, RandomAgent
from harness.neural import (
    AlphaZeroAgent,
    Connect4Net,
    build_alphazero_agent,
    encode,
    head_to_head,
    load_net,
    save_net,
    self_play_game,
    train_alphazero,
    train_net,
    vs_opponent_game,
)


def _game():
    return Connect4()


def _eval_loss(net, data):
    x = torch.stack([e[0] for e in data])
    tp = torch.tensor([e[1] for e in data], dtype=torch.float32)
    tv = torch.tensor([[e[2]] for e in data], dtype=torch.float32)
    net.eval()
    with torch.no_grad():
        logits, value = net(x)
        loss = -(tp * F.log_softmax(logits, dim=1)).sum(1).mean() + F.mse_loss(value, tv)
    return float(loss)


def test_encode_shape_and_side_to_move_perspective():
    game = _game()
    s = game.initial_state(random.Random(0))
    x = encode(game, s)
    assert tuple(x.shape) == (2, 6, 7)
    assert float(x.sum()) == 0.0  # empty board
    s = game.step(s, 3)  # player 0 drops in column 3; now player 1 is to move
    x = encode(game, s)
    assert float(x[0].sum()) == 0.0  # side-to-move (p1) has no pieces yet
    assert float(x[1].sum()) == 1.0  # the opponent's piece sits in the opponent plane


def test_net_forward_shapes_and_value_range():
    net = Connect4Net()
    logits, value = net(torch.zeros(4, 2, 6, 7))
    assert tuple(logits.shape) == (4, 7)
    assert tuple(value.shape) == (4, 1)
    assert -1.0 <= float(value.min()) and float(value.max()) <= 1.0


def test_alphazero_agent_plays_legal_and_finds_an_immediate_win():
    game = _game()
    torch.manual_seed(0)
    net = Connect4Net()
    s = game.initial_state(random.Random(0))
    assert AlphaZeroAgent(net, sims=40).act(game, s, random.Random(0)) in game.legal_actions(s)
    # player 0 has three in column 3 and is to move — the net-guided search must find the terminal win.
    for c in [3, 0, 3, 1, 3, 2]:
        s = game.step(s, c)
    assert AlphaZeroAgent(net, sims=150).act(game, s, random.Random(0)) == 3


def test_self_play_returns_training_examples():
    game = _game()
    ex = self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=20), random.Random(0))
    assert len(ex) > 0
    x, pi, z = ex[0]
    assert tuple(x.shape) == (2, 6, 7)
    assert abs(sum(pi) - 1.0) < 1e-4
    assert z in (-1.0, 0.0, 1.0)


def test_training_reduces_loss_on_self_play_data():
    game = _game()
    torch.manual_seed(0)
    data = []
    for i in range(5):
        data += self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=20), random.Random(i))
    net = Connect4Net()
    torch.manual_seed(0)
    before = _eval_loss(net, data)
    train_net(net, data, epochs=10, batch_size=32, lr=1e-2, device="cpu")
    after = _eval_loss(net, data)
    assert after < before  # the net learns to fit its own self-play targets


def test_save_load_weights_roundtrip(tmp_path):
    torch.manual_seed(1)
    net = Connect4Net()
    path = str(tmp_path / "w.pt")
    save_net(net, path)
    net2 = load_net(path)
    x = torch.zeros(1, 2, 6, 7)
    l1, v1 = net(x)
    l2, v2 = net2(x)
    assert torch.allclose(l1, l2) and torch.allclose(v1, v2)


def test_build_alphazero_agent_from_saved_weights(tmp_path):
    net = Connect4Net()
    path = str(tmp_path / "w.pt")
    save_net(net, path)
    agent = build_alphazero_agent({"az_weights": path, "az_sims": 10})
    game = _game()
    a = agent.act(game, game.initial_state(random.Random(0)), random.Random(0))
    assert a in range(7)


def test_resolve_agent_routes_alphazero(tmp_path):
    from harness.agents import resolve_agent

    path = str(tmp_path / "w.pt")
    save_net(Connect4Net(), path)
    agent = resolve_agent("alphazero", {"az_weights": path, "az_sims": 8})
    assert isinstance(agent, AlphaZeroAgent)


def test_vs_opponent_game_collects_only_learner_moves():
    game = _game()
    learner = AlphaZeroAgent(Connect4Net(), sims=15)
    ex = vs_opponent_game(game, learner, RandomAgent(), learner_seat=0, rng=random.Random(0))
    assert len(ex) > 0
    x, pi, z = ex[0]
    assert tuple(x.shape) == (2, 6, 7)
    assert abs(sum(pi) - 1.0) < 1e-4 and z in (-1.0, 0.0, 1.0)


def test_train_alphazero_warm_starts_from_init_net():
    game = _game()
    torch.manual_seed(0)
    init = Connect4Net()
    net, _ = train_alphazero(game, iterations=1, selfplay_games=2, sims=10, epochs=1, init_net=init, seed=0)
    assert net is init  # warm-start trains the passed-in champion in place, it is not replaced by a fresh net


def test_train_alphazero_uses_the_opponent_pool():
    game = _game()
    net, hist = train_alphazero(
        game, iterations=1, selfplay_games=6, sims=10, epochs=1, opponent_pool=[HeuristicAgent], pool_frac=1.0, seed=0
    )
    assert hist[0]["vs_pool_games"] == 6  # pool_frac=1.0 → every game is against the pool


def test_head_to_head_reports_normalised_rates():
    game = _game()
    r = head_to_head(game, RandomAgent, RandomAgent, n=6, rng=random.Random(0))
    assert set(r) == {"win_rate", "draw_rate", "loss_rate", "games"}
    assert abs(r["win_rate"] + r["draw_rate"] + r["loss_rate"] - 1.0) < 1e-9


def test_head_to_head_diversifies_games_via_random_openings():
    game = _game()
    net = Connect4Net()  # played against ITSELF: without random openings every greedy game would be identical
    r = head_to_head(
        game, lambda: AlphaZeroAgent(net, sims=10), lambda: AlphaZeroAgent(net, sims=10), n=20, rng=random.Random(0)
    )
    assert 0.0 < r["win_rate"] < 1.0  # random openings make the n games genuinely distinct
