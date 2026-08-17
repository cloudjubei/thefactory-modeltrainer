import random

import torch
import torch.nn.functional as F

from games.connect4 import Connect4
from harness.agents import HeuristicAgent, RandomAgent
from harness.neural import (
    AlphaZeroAgent,
    Connect4Net,
    augment_examples,
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


def test_augment_examples_mirrors_board_and_policy():
    g = Connect4()
    x = torch.zeros(2, 6, 7)
    x[0, 0, 0] = 1.0  # own stone in column 0
    aug = augment_examples([(x, [1.0, 0, 0, 0, 0, 0, 0], 1.0)], g.symmetries())
    assert len(aug) == 2  # identity + mirror
    mx, mpi, mv = aug[1]
    assert mv == 1.0
    assert mpi == [0, 0, 0, 0, 0, 0, 1.0]  # policy mass reflected to column 6
    assert float(mx[0, 0, 6]) == 1.0 and float(mx[0, 0, 0]) == 0.0


def test_augment_examples_identity_only_is_copy():
    x = torch.zeros(2, 6, 7)
    assert len(augment_examples([(x, [0.0] * 7, 0.0)], [[0, 1, 2, 3, 4, 5, 6]])) == 1


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


def test_alphazero_solve_endgame_plays_perfectly_on_late_positions():
    from harness.benchmark import sample_solvable_positions
    from harness.solver import optimal_columns

    game = _game()
    torch.manual_seed(0)
    # An UNTRAINED net would blunder late positions; with the exact-endgame cutoff on, greedy play is
    # game-theoretically optimal regardless of the net — the solver, not the value head, chooses the endgame.
    agent = AlphaZeroAgent(Connect4Net(), sims=1, solve_endgame=40)
    states = sample_solvable_positions(game, n=15, min_moves=30, seed=5)
    assert states
    for s in states:
        assert agent.act(game, s, random.Random(0)) in optimal_columns(s)


def test_alphazero_solve_endgame_off_by_default_and_skipped_while_exploring():
    game = _game()
    # Off by default (self-play must keep exploring, not collapse onto solver moves).
    assert AlphaZeroAgent(Connect4Net(), sims=5).solve_endgame == 0
    # Even with the cutoff on, a non-zero temperature (self-play exploration) skips it — the solver is a
    # GREEDY-play optimisation, never a training-time shortcut that would starve exploration.
    agent = AlphaZeroAgent(Connect4Net(), sims=5, solve_endgame=40, temperature=1.0)
    s = game.initial_state(random.Random(0))
    for c in [3, 3, 3, 4]:  # a few stones down, still far from a cheap solve anyway
        s = game.step(s, c)
    assert agent.act(game, s, random.Random(0)) in game.legal_actions(s)  # plays via search, not a crash


def test_build_alphazero_agent_reads_solve_endgame(tmp_path):
    save_net(Connect4Net(), str(tmp_path / "w.pt"))
    agent = build_alphazero_agent({"az_weights": str(tmp_path / "w.pt"), "az_sims": 8, "az_solve_endgame": 14})
    assert agent.solve_endgame == 14


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


def test_oracle_distill_games_labels_the_opening_including_centre_first():
    from harness.neural import oracle_distill_games

    game = Connect4()
    ex = oracle_distill_games(game, n_games=4, seed=0, oracle_depth=6, exact_max_empty=10)
    assert ex
    for x, pi, z in ex:
        assert tuple(x.shape) == (2, 6, 7)
        assert abs(sum(pi) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)
    # game 0 has the learner as the FIRST player, so the very first labelled example is the EMPTY board — and its
    # optimal label must be CENTRE (col 3), the move the champion's broken opening failed to play.
    x0, pi0, _ = ex[0]
    assert float(x0.sum()) == 0.0  # empty board
    assert max(range(7), key=lambda c: pi0[c]) == 3  # centre-first


def test_build_distill_corpus_caches_to_disk(tmp_path):
    from harness.neural import build_distill_corpus

    game = Connect4()
    spec = {"games": 4, "seed": 1, "oracle_depth": 6, "exact_max_empty": 10, "late": {"n": 3, "min_moves": 28}}
    a = build_distill_corpus(game, spec, cache_dir=str(tmp_path))
    assert a and len(list(tmp_path.glob("*.pt"))) == 1  # a cache file was written
    b = build_distill_corpus(game, spec, cache_dir=str(tmp_path))  # second call HITS the cache
    assert len(a) == len(b)


def test_distill_examples_targets_match_the_oracle():
    from harness.benchmark import sample_solvable_positions
    from harness.neural import distill_examples
    from harness.solver import optimal_columns

    game = Connect4()
    n, min_moves, seed = 6, 28, 3  # late-game → fast exact solves
    states = sample_solvable_positions(game, n, min_moves, seed)
    examples = distill_examples(game, n, min_moves, seed)
    assert examples and len(examples) == len(states)
    for (x, pi, value), state in zip(examples, states):
        assert x.shape == (2, 6, 7)
        optimal = optimal_columns(state)
        assert sorted(i for i, p in enumerate(pi) if p > 0) == optimal  # mass exactly on the optimal set
        assert all(abs(pi[c] - 1.0 / len(optimal)) < 1e-6 for c in optimal)  # uniform over it
        assert value in (-1.0, 0.0, 1.0)
