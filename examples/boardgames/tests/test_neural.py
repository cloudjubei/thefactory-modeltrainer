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
    completed_q_policy,
    completed_q_values,
    encode,
    head_to_head,
    load_net,
    oracle_distill_games,
    save_net,
    self_play_game,
    train_alphazero,
    train_net,
    vs_opponent_game,
)
from harness.solver import optimal_columns
from harness.tablebase import Tablebase


def test_alphazero_proven_value_reads_the_book_and_the_solvable_endgame():
    from games.connect4 import Connect4
    from harness.benchmark import sample_solvable_positions
    from harness.solver import move_values
    from harness.tablebase import Tablebase

    game = Connect4()
    torch.manual_seed(0)
    late = sample_solvable_positions(game, n=1, min_moves=30, seed=2)[0]
    proven = max(move_values(late, weak=True).values())  # the mover's exact value here

    book = Tablebase(cap=100)
    book.put_proven(game.canonical_key(late), proven)
    assert AlphaZeroAgent(Connect4Net(), sims=1, book=book)._proven_value(game, late) == float(proven)
    # no book but a solvable endgame → exact solve value (the search backs up truth, not the net's estimate)
    assert AlphaZeroAgent(Connect4Net(), sims=1, solve_endgame=40)._proven_value(game, late) is not None
    # no book, no solve_endgame (self-play default) → None, so exploration/self-play is unchanged
    assert AlphaZeroAgent(Connect4Net(), sims=1)._proven_value(game, late) is None


def test_proven_value_write_through_memoizes_once():
    # §C.7 #2: with BOTH a book AND solve_endgame, a cheap-solvable endgame is solved ONCE and written back
    # (value-only), so a second visit is a free lookup — the amortisation that makes the online loop pay off.
    from games.connect4 import Connect4
    from harness.benchmark import sample_solvable_positions
    from harness.book import _key
    from harness.tablebase import Tablebase

    game = Connect4()
    late = sample_solvable_positions(game, n=1, min_moves=30, seed=5)[0]
    book = Tablebase(cap=100)
    agent = AlphaZeroAgent(Connect4Net(), sims=1, book=book, solve_endgame=40)
    assert agent.endgame_solves == 0 and agent.endgame_hits == 0

    v1 = agent._proven_value(game, late)  # first visit: solves + writes through
    assert v1 is not None
    key = _key(game, late)
    assert book.is_proven(key)  # the solve was recorded into the run tablebase
    assert agent.endgame_solves == 1 and agent.endgame_hits == 0
    assert book.entry(key).best_actions == 0  # VALUE-ONLY: no best_actions (mirror column-flip bug is dodged)

    v2 = agent._proven_value(game, late)  # second visit: pure lookup, no re-solve
    assert v2 == v1
    assert agent.endgame_solves == 1 and agent.endgame_hits == 1


def test_exact_targets_override_selfplay_value_sign():
    # §C.7 #2: a proven value in the endgame tablebase REPLACES the self-play value target for that position,
    # mover-relative direct-assign (NO negation — canonical_key and outcome_for are both side-to-move-relative).
    from games.connect4 import Connect4
    from harness.book import _key
    from harness.tablebase import Tablebase

    game = Connect4()
    torch.manual_seed(0)
    agent = AlphaZeroAgent(Connect4Net(), sims=8)  # NO book on the agent → search/trajectory unaffected by the pin
    base = self_play_game(game, agent, random.Random(11), return_states=True)  # (state, x, pi, v)
    assert base, "self-play produced no examples"
    j = len(base) - 1  # a near-terminal pending position
    state_j, _x, _pi, v_nat = base[j]
    sentinel = -1 if v_nat >= 0 else 1  # opposite sign + nonzero → guards a spurious negation

    tb = Tablebase(cap=100)
    tb.put_proven(_key(game, state_j), sentinel)
    # reuse the SAME agent (identical net weights) + reseed the SAME rng → the search-driving randomness, and thus
    # the trajectory, is reproduced exactly; the pin only rewrites the value TARGET, never the play.
    over = self_play_game(game, agent, random.Random(11),
                          return_states=True, endgame_tb=tb, exact_value_targets=True)
    by_key = {_key(game, s): v for (s, _x, _pi, v) in over}
    assert by_key[_key(game, state_j)] == float(sentinel)  # the booked position's target was overridden, sign intact
    # a DIFFERENT, unbooked position keeps its natural target (targeted override, not a blanket relabel)
    if len(base) >= 2:
        state_0 = base[0][0]
        if _key(game, state_0) != _key(game, state_j):
            assert by_key[_key(game, state_0)] == base[0][3]


def test_mix_training_set_holds_the_anchor_at_a_fixed_fraction():
    from harness.neural import _mix_training_set

    buffer = list(range(8000))
    distilled = list(range(400))  # plain concatenation → 400/8400 ≈ 4.8% (dilutes to noise as the buffer fills)
    mixed = _mix_training_set(buffer, distilled, 0.34)
    anchor_frac = (len(mixed) - len(buffer)) / len(mixed)
    assert 0.28 < anchor_frac < 0.40  # the exact-play anchor is held near its target, not diluted
    assert _mix_training_set(buffer, distilled, 0.0) == buffer + distilled  # 0 = the old plain concatenation
    assert _mix_training_set([], distilled, 0.34) == distilled  # no self-play yet → just the anchor
    assert _mix_training_set(buffer, [], 0.34) == buffer  # no anchor → the buffer unchanged


def test_oracle_distillation_relabels_the_value_target_from_the_proven_book():
    # The value-label contamination fix: where the book PROVES a position's value, distillation uses THAT (exact,
    # mover-relative) instead of the noisy self-play OUTCOME — so a proven opening keeps its true value even when
    # the labelled game happened to end differently.
    game = Connect4()
    book = Tablebase(cap=100)
    empty = game.initial_state(random.Random(0))
    book.put_proven(game.canonical_key(empty), 0)  # a distinctive PROVEN opening value (draw) the outcome won't be
    ex = oracle_distill_games(game, n_games=2, seed=0, oracle_depth=4, book=book)  # shallow oracle → fast
    assert ex[0][2] == 0.0  # first learner example = the empty board → its value target is the book's proof


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


def test_two_hot_targets_project_onto_the_support():
    # §C.8 #3: a scalar target in [-1,1] becomes a two-hot distribution over the bin support — mass sums to 1,
    # an exact bin centre gets ALL of it, and an off-centre value splits linearly between its two neighbours.
    from harness.neural import _two_hot

    support = torch.linspace(-1.0, 1.0, 5)  # centres at -1, -0.5, 0, 0.5, 1
    d = _two_hot(torch.tensor([[0.0]]), support)
    assert d.shape == (1, 5)
    assert float(d.sum()) == 1.0 and float(d[0, 2]) == 1.0
    d = _two_hot(torch.tensor([[0.25]]), support)  # halfway between 0 and 0.5
    assert abs(float(d[0, 2]) - 0.5) < 1e-6 and abs(float(d[0, 3]) - 0.5) < 1e-6
    d = _two_hot(torch.tensor([[-1.0], [1.0]]), support)  # clamped edges stay valid one-hots
    assert float(d[0, 0]) == 1.0 and float(d[1, 4]) == 1.0


def test_training_reduces_loss_with_a_categorical_value_head():
    # The bins path must LEARN through the same train_net entry (cross-entropy on the bin logits) and the
    # deployed scalar expectation must move toward the target sign — no consumer sees the distribution.
    game = _game()
    torch.manual_seed(0)
    data = []
    for i in range(3):
        data += self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=10), random.Random(i))
    net = Connect4Net(channels=16, blocks=1, residual=True, batchnorm=False, head_hidden=8, value_bins=11)
    torch.manual_seed(0)
    first = train_net(net, data, epochs=1, batch_size=32, lr=1e-2, device="cpu")
    last = train_net(net, data, epochs=10, batch_size=32, lr=1e-2, device="cpu")
    assert last < first  # loss falls through the categorical path
    net.eval()
    won = [e for e in data if e[2] == 1.0][:8]
    with torch.no_grad():
        vals = [float(net(e[0].unsqueeze(0))[1][0, 0]) for e in won]
    assert sum(vals) / max(1, len(vals)) > 0.0  # expectation tracks the +1 targets it was trained on


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


def test_build_alphazero_agent_defaults_the_endgame_cutoff_on(tmp_path):
    from harness.config import DEFAULT_AZ_SOLVE_ENDGAME

    save_net(Connect4Net(), str(tmp_path / "w.pt"))
    # A deployed agent (no explicit az_solve_endgame in its spec — e.g. a champion crowned before the default
    # changed) must still play the endgame exactly, so the registry seam defaults the cutoff ON.
    agent = build_alphazero_agent({"az_weights": str(tmp_path / "w.pt"), "az_sims": 8})
    assert agent.solve_endgame == DEFAULT_AZ_SOLVE_ENDGAME > 0


def test_net_value_is_a_side_to_move_scalar_in_range():
    from harness.neural import net_value

    game = _game()
    torch.manual_seed(0)
    v = net_value(Connect4Net(), game, game.initial_state(random.Random(0)))
    assert isinstance(v, float) and -1.0 <= v <= 1.0  # the opening value-belief gauge (should → +1 once trained)


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


# --- MCTS-Solver: proof PROPAGATION half in the net-guided core (greedy deployment only) -------------------
def _c4_win(game, empties):
    """The first sampled connect4 position with exactly `empties` empty cells that is a proven WIN for the mover
    and NOT a mate-in-1 — so the search must DERIVE a proof (children solve, the win propagates to the root)."""
    from harness.benchmark import sample_solvable_positions

    target_ply = 42 - empties
    for seed in range(200):
        for s in sample_solvable_positions(game, n=4, min_moves=target_ply, seed=seed):
            if sum(1 for v in s.board if v != 0) != target_ply:
                continue
            me = game.current_player(s)
            if any(game.step(s, a).winner == me for a in game.legal_actions(s)):
                continue
            if game.position_value(s) == 1:
                return s
    return None


def test_alphazero_solver_proves_the_root_and_plays_the_winning_move():
    from harness.agents import state_key
    from harness.solver import optimal_columns

    game = _game()
    torch.manual_seed(0)
    pos = _c4_win(game, empties=12)  # a proven win, NOT a mate-in-1
    assert pos is not None
    empties = sum(1 for v in pos.board if v == 0)
    # An UNTRAINED net would blunder here; but the root is NOT directly solvable (12 empty > solve_endgame) while its
    # children ARE — greedy play must derive the proof by propagation and play the win, not trust the value head.
    agent = AlphaZeroAgent(Connect4Net(), sims=80, solve_endgame=empties - 1)
    move = agent.act(game, pos, random.Random(0))
    assert agent._proven[state_key(game, pos)] == 1.0  # the root became a DERIVED proof
    assert move in optimal_columns(pos)  # and greedy play is game-theoretically optimal


def test_alphazero_solver_leaves_self_play_policy_intact():
    # Self-play safety: the overlay must NOT prune the tree or stop the search early, or the visit-count policy π
    # (the training target) would be distorted. Even WITH a solver active, run_search spends the full budget and
    # returns a policy over every legal move.
    game = _game()
    torch.manual_seed(0)
    pos = _c4_win(game, empties=12)
    assert pos is not None
    empties = sum(1 for v in pos.board if v == 0)
    agent = AlphaZeroAgent(Connect4Net(), sims=50, solve_endgame=empties - 1, temperature=1.0, add_noise=True)
    pi = agent.run_search(game, pos, random.Random(0))
    assert set(pi) == set(game.legal_actions(pos))  # π covers all legal moves (no pruning collapsed it)
    assert abs(sum(pi.values()) - 1.0) < 1e-9
    assert agent.sims_used == 50  # the full search budget ran — no early proof termination in self-play


def test_alphazero_solver_off_without_book_or_solver_stays_pure():
    from harness.agents import state_key

    game = _game()
    torch.manual_seed(0)
    pos = _c4_win(game, empties=12)
    assert pos is not None
    agent = AlphaZeroAgent(Connect4Net(), sims=60)  # pure net: no book, solve_endgame 0
    agent.act(game, pos, random.Random(0))
    assert agent._proven == {}  # the solver overlay is inert without a proof source
    assert state_key  # (imported for symmetry with the sibling tests)


# --- book → net: best_actions / estimate value as SOFT distillation targets --------------------------------
def test_book_distill_examples_weights_proofs_over_estimates():
    from harness.neural import book_distill_examples

    game = _game()
    torch.manual_seed(0)
    book = Tablebase(cap=100)
    a = game.initial_state(random.Random(0))  # an ESTIMATE: a soft belief (value 0.4, one best move)
    b = game.step(a, 3)  # a PROVEN entry: exact (value 1.0, two optimal moves)
    book.put_estimate(game.canonical_key(a), 0.4, best_actions=(1 << 3), n=5)
    book.put_proven(game.canonical_key(b), 1, best_actions=(1 << 3) | (1 << 4))
    ex = book_distill_examples(game, book, [a, b], proof_copies=3, estimate_copies=1)
    est = [e for e in ex if abs(e[2] - 0.4) < 1e-6]
    prf = [e for e in ex if e[2] == 1.0]
    assert len(est) == 1 and len(prf) == 3  # a proof outweighs a belief by whole-copy replication (no loss weights)
    assert abs(est[0][1][3] - 1.0) < 1e-6  # estimate policy: all mass on its single best move (soft value kept: 0.4)
    assert abs(prf[0][1][3] - 0.5) < 1e-6 and abs(prf[0][1][4] - 0.5) < 1e-6  # proof policy: uniform over the set


def test_book_distill_examples_skips_uncovered_and_actionless_positions():
    from harness.neural import book_distill_examples

    game = _game()
    torch.manual_seed(0)
    book = Tablebase(cap=100)
    covered = game.initial_state(random.Random(0))
    uncovered = game.step(covered, 0)  # never booked → skipped
    book.put_proven(game.canonical_key(covered), 0, best_actions=0)  # booked but NO best_actions → skipped
    assert book_distill_examples(game, book, [covered, uncovered]) == []


def test_train_alphazero_folds_in_book_distillation():
    from harness.benchmark import sample_solvable_positions

    game = _game()
    torch.manual_seed(0)
    states = sample_solvable_positions(game, 4, 6, seed=0)
    book = Tablebase(cap=100)
    for s in states:  # give each sampled opening position a book belief so distillation covers it
        book.put_estimate(game.canonical_key(s), 0.2, best_actions=(1 << game.legal_actions(s)[0]), n=3)
    net, hist = train_alphazero(
        game, iterations=1, selfplay_games=1, sims=4, epochs=1, seed=0,
        book=book, book_distill_positions=4, book_distill_min_moves=6,
    )
    assert isinstance(net, Connect4Net) and len(hist) == 1  # ran with book distillation folded into the anchor


# --- GUMBEL + COMPLETED-Q POLICY TARGET (plan §C.5 priority #1) -----------------------------------------------
def _win_in_1_root(seed: int):
    """A non-terminal Connect 4 position where the side to move has an IMMEDIATE winning move and ≥2 legal
    replies — a tactical position whose optimal set the exact solver knows, and where terminal backup gives the
    winning move Q=+1 regardless of the net, so the improvement is attributable to the OPERATOR, not a trained net."""
    game = Connect4()
    rng = random.Random(seed)
    for _ in range(40000):
        s = game.initial_state(rng)
        while not s.done:
            legal = game.legal_actions(s)
            if len(legal) >= 2 and any(game.step(s, a).winner == s.to_move for a in legal):
                return s
            s = game.step(s, rng.choice(legal))
    raise AssertionError("no win-in-1 root found — widen the search")


def test_completed_q_values_uses_v_mix_for_unvisited_actions():
    # An unvisited action is COMPLETED with v_mix (the visited siblings' prior-weighted Q blended with the root
    # value), NOT an arbitrary 0 — the rule that keeps the target sane when few actions were visited at low sims.
    prior = {0: 0.5, 1: 0.5}
    child_n = {0: 0, 1: 4}
    child_w = {0: 0.0, 1: 4.0}  # action 1 visited 4× with mean Q = +1
    q, sum_n, max_n = completed_q_values(prior, child_n, child_w, root_value=0.0, legal=[0, 1])
    assert q[1] == 1.0  # a visited action keeps its OWN mean Q
    # v_mix = (root_value + sum_n · (Σπ·Q / Σπ_visited)) / (1 + sum_n) = (0 + 4·(0.5·1 / 0.5)) / 5 = 0.8
    assert abs(q[0] - 0.8) < 1e-9
    assert sum_n == 4 and max_n == 4


def test_completed_q_policy_is_a_policy_improvement_over_a_bad_prior():
    # THE GUARANTEE (Danihelka 2022): given Q that clearly favours action 2 but a prior that wrongly favours
    # action 0, the completed-Q policy shifts mass ONTO the high-Q action — more than the prior, and more than the
    # (uniform) visit-count policy would at low sims. This is the σ / v_mix correctness gate the plan requires.
    prior = {0: 0.8, 1: 0.1, 2: 0.1}
    child_n = {0: 2, 1: 2, 2: 2}          # equal visits → the raw visit-count policy is uniform (1/3 each)
    child_w = {0: -2.0, 1: 0.0, 2: 2.0}   # Q: 0 → -1, 1 → 0, 2 → +1
    pi = completed_q_policy(prior, child_n, child_w, root_value=0.0, legal=[0, 1, 2])
    assert abs(sum(pi.values()) - 1.0) < 1e-9 and all(p >= 0.0 for p in pi.values())
    assert max(pi, key=pi.get) == 2   # picks the high-Q action, NOT the prior's wrong favourite
    assert pi[2] > prior[2]           # improvement OVER the prior
    assert pi[2] > 1 / 3              # BEATS the uniform visit-count policy at low sims
    assert pi[0] < prior[0]           # and demotes the prior's wrong favourite


def test_gumbel_search_returns_a_valid_improved_policy_within_budget():
    game = Connect4()
    root = _win_in_1_root(seed=1)
    torch.manual_seed(0)
    agent = AlphaZeroAgent(Connect4Net(), sims=8, gumbel=True)
    agent._nodes = {}
    pi = agent.run_search(game, root, random.Random(0))
    legal = game.legal_actions(root)
    assert set(pi) == set(legal) and abs(sum(pi.values()) - 1.0) < 1e-6 and all(p >= 0.0 for p in pi.values())
    assert agent.sims_used <= 8  # an HONEST budget: no more simulations than a raw n=8 search
    wins = [a for a in legal if game.step(root, a).winner == root.to_move]
    assert agent._gumbel_selected in wins  # guaranteed improvement + terminal backup ⇒ it selects the win


def test_completed_q_policy_beats_visit_counts_at_low_sims_on_tactical_positions():
    # THE #1 GATE (plan §C.5): at n=8 sims the completed-Q policy concentrates MORE mass on the exact-optimal move
    # than the raw visit-count policy across a set of tactical Connect-4 positions — measured against the solver,
    # with NO trained net (a fresh net), so the gain is attributable to the operator. Trust it only once this holds.
    game = Connect4()
    roots = [_win_in_1_root(seed=s) for s in range(1, 9)]
    torch.manual_seed(0)
    net = Connect4Net()
    q_mass = v_mass = 0.0
    gumbel_hits = visit_hits = 0
    for r in roots:
        opt = set(optimal_columns(r))
        va = AlphaZeroAgent(net, sims=8, gumbel=False)
        va._nodes = {}
        vpi = va.run_search(game, r, random.Random(0))
        ga = AlphaZeroAgent(net, sims=8, gumbel=True)
        ga._nodes = {}
        gpi = ga.run_search(game, r, random.Random(0))
        v_mass += sum(vpi.get(a, 0.0) for a in opt)
        q_mass += sum(gpi.get(a, 0.0) for a in opt)
        visit_hits += max(vpi, key=vpi.get) in opt
        gumbel_hits += max(gpi, key=gpi.get) in opt
    assert q_mass > v_mass          # completed-Q puts MORE mass on optimal — it BEATS the visit-count target
    assert gumbel_hits >= visit_hits  # and never argmaxes the optimal move LESS often than visit counts
    assert gumbel_hits == len(roots)  # in fact it converts every immediate win (the improvement guarantee)


def test_self_play_with_gumbel_plays_the_selected_action_and_stores_completed_q_targets():
    # Gumbel self-play plays the Sequential-Halving winner (`_gumbel_selected`) and stores the completed-Q
    # improved policy as the target — no Dirichlet noise (Gumbel supplies the root exploration itself).
    game = Connect4()
    torch.manual_seed(0)
    agent = AlphaZeroAgent(Connect4Net(), sims=8, gumbel=True)
    ex = self_play_game(game, agent, random.Random(0))
    assert agent.add_noise is False  # Gumbel replaces Dirichlet — no double exploration
    assert len(ex) > 0
    x, pi, z = ex[0]
    assert tuple(x.shape) == (2, 6, 7) and abs(sum(pi) - 1.0) < 1e-4 and z in (-1.0, 0.0, 1.0)


def test_gumbel_search_is_deterministic_and_noise_free_when_greedy():
    # Gumbel noise is a SELF-PLAY exploration device; at greedy eval (temperature 0) it must be OFF, so greedy
    # deployment is deterministic and doesn't self-sabotage by spreading root visits. Two greedy searches with
    # DIFFERENT rng streams must agree on both the policy and the selected move.
    game = Connect4()
    root = _win_in_1_root(seed=3)
    torch.manual_seed(0)
    net = Connect4Net()
    a = AlphaZeroAgent(net, sims=8, gumbel=True, temperature=0.0)
    a._nodes = {}
    pa = a.run_search(game, root, random.Random(1))
    b = AlphaZeroAgent(net, sims=8, gumbel=True, temperature=0.0)
    b._nodes = {}
    pb = b.run_search(game, root, random.Random(999))  # a different rng stream
    assert a._gumbel_selected == b._gumbel_selected  # noise-free ⇒ rng-independent selection
    assert pa == pb  # and an identical completed-Q target


def test_completed_q_policy_preserves_q_magnitude_soft_on_ties_peaky_on_decisive():
    # The calibration GUARD (the σ/normalisation fix): the completed-Q target must scale with the CARDINAL Q gap,
    # not its rank. Near-tied Q's stay a SOFT target (no manufactured confidence); a decisive win/loss gap peaks.
    # This fails under per-root min-max (which stretches any nonzero gap to full [0,1] → near-one-hot on noise).
    near = completed_q_policy({0: 0.5, 1: 0.5}, {0: 4, 1: 4}, {0: 2.80, 1: 2.88}, 0.0, [0, 1])  # Q 0.70 vs 0.72
    assert 0.2 < near[0] < 0.8 and 0.2 < near[1] < 0.8  # near-tied ⇒ neither action collapses to ~0/~1
    decisive = completed_q_policy({0: 0.5, 1: 0.5}, {0: 4, 1: 4}, {0: -4.0, 1: 4.0}, 0.0, [0, 1])  # Q -1 vs +1
    assert decisive[1] > 0.9  # a genuine win/loss gap ⇒ the target peaks on the winning move


def test_n_step_value_targets_bootstrap_from_target_net_else_terminal_outcome():
    # #3: the n-step value target replaces the raw-MC outcome (contaminated by LATER blunders) with the lagged
    # target-net value n plies ahead, sign-corrected to the mover (n even → same mover, n odd → opponent), and
    # falls back to the real terminal outcome only when the terminal is within n plies.
    from harness.neural import n_step_value_targets

    vt = [0.1, -0.2, 0.3, -0.4, 0.5]        # lagged target-net values, mover-relative, 5 pending positions
    outcome = [1.0, -1.0, 1.0, -1.0, 1.0]   # P0 wins → mover-relative MC outcomes alternate
    t2 = n_step_value_targets(vt, outcome, n=2)  # even → sign +1
    assert t2[0] == 0.3 and t2[1] == -0.4 and t2[2] == 0.5      # bootstrap vt[i+2]
    assert t2[3] == -1.0 and t2[4] == 1.0                       # terminal within 2 → real outcome
    t1 = n_step_value_targets(vt, outcome, n=1)  # odd → sign −1
    assert abs(t1[0] - 0.2) < 1e-9 and abs(t1[3] - (-0.5)) < 1e-9  # bootstrap −vt[i+1]
    assert t1[4] == 1.0                                          # last position: terminal next ply
    # n large enough that EVERY position's terminal is within n → pure MC (no bootstrap), i.e. current behaviour
    assert n_step_value_targets(vt, outcome, n=5) == outcome


def test_train_alphazero_with_n_step_value_target_and_lagged_net_runs():
    # #3 integration: the loop runs with the n-step value target off a lagged target net (refreshed every k iters),
    # producing a net and history — the value label now bootstraps off the target net, not the raw final outcome.
    game = Connect4()
    net, hist = train_alphazero(
        game, iterations=3, selfplay_games=2, sims=4, epochs=1, seed=0,
        gumbel=True, value_n_step=6, target_refresh=2,
    )
    assert isinstance(net, Connect4Net) and len(hist) == 3


def test_reanalyze_examples_relabels_states_with_the_current_net():
    # #2 Reanalyze: re-running the CURRENT net's search over stored states regenerates fresh (policy, value)
    # targets — a valid distribution + a bounded search-improved value — so old buffer entries are de-staled for
    # ~free (no new self-play). Deterministic (greedy, seeded).
    from harness.neural import reanalyze_examples

    game = Connect4()
    torch.manual_seed(0)
    net = Connect4Net()
    agent = AlphaZeroAgent(net, sims=16, gumbel=True, temperature=0.0)
    states = []
    for s in range(3):  # three distinct mid-game states from short random playouts
        st = game.initial_state(random.Random(s))
        for _ in range(4 + s):
            st = game.step(st, random.Random(s).choice(game.legal_actions(st)))
        states.append(st)
    ex = reanalyze_examples(game, agent, states, random.Random(0))
    assert len(ex) == len(states)
    for x, pi, v in ex:
        assert tuple(x.shape) == (2, 6, 7)
        assert abs(sum(pi) - 1.0) < 1e-5 and all(p >= 0.0 for p in pi)
        assert -1.0 <= v <= 1.0


def test_train_alphazero_with_reanalyze_runs_and_relabels_the_buffer():
    # #2 integration: the loop runs with a state-carrying buffer and re-labels a fraction of it with the current
    # net each iteration (history records the reanalyzed count), on top of the n-step value target + gumbel search.
    game = Connect4()
    net, hist = train_alphazero(
        game, iterations=3, selfplay_games=2, sims=4, epochs=1, seed=0,
        gumbel=True, value_n_step=6, target_refresh=2, reanalyze_frac=0.5,
    )
    assert isinstance(net, Connect4Net) and len(hist) == 3
    assert any(h["reanalyzed"] > 0 for h in hist)  # at least one later iteration re-labelled buffer entries


def test_self_play_opening_plies_trains_from_diverse_off_line_positions():
    # The ROBUSTNESS lever: with opening_plies>0, self-play's first RECORDED position is already off the canonical
    # start (the random opening's stones are down), so the net trains on off-main-line positions — the fix for a
    # net that plays the main line perfectly but LOSES away from it. The random plies themselves aren't recorded.
    game = Connect4()
    torch.manual_seed(0)
    agent = AlphaZeroAgent(Connect4Net(), sims=6, gumbel=True)
    ex0 = self_play_game(game, agent, random.Random(0), opening_plies=0)
    assert float(ex0[0][0].sum()) == 0.0  # canonical: first recorded position is the empty board
    ex4 = self_play_game(game, agent, random.Random(0), opening_plies=4)
    assert float(ex4[0][0].sum()) == 4.0  # diverse: first recorded position already carries the 4 opening stones


def test_train_alphazero_buffer_persists_across_calls():
    # §C.7 batched training must be equivalent to a continuous run: return the replay buffer and accept it back so
    # a resumed batch does NOT restart from an empty buffer (which would starve a big net that needs more data).
    from games.connect4 import Connect4
    game = Connect4()
    net, hist, buf = train_alphazero(game, iterations=1, selfplay_games=4, sims=6, epochs=1, seed=0,
                                     buffer_cap=5000, return_buffer=True)
    assert isinstance(buf, list) and len(buf) > 0  # the accumulated (x, pi, v) examples come back
    # Seed a fresh call with that buffer: after 1 iter it must hold MORE than a from-empty call (init + fresh).
    _n2, _h2, buf_seeded = train_alphazero(game, iterations=1, selfplay_games=4, sims=6, seed=1,
                                           buffer_cap=50000, init_buffer=buf, return_buffer=True)
    _n3, _h3, buf_fresh = train_alphazero(game, iterations=1, selfplay_games=4, sims=6, seed=1,
                                          buffer_cap=50000, return_buffer=True)
    assert len(buf_seeded) > len(buf_fresh)  # the persisted buffer carried forward


def test_selfplay_worker_reproduces_single_game(tmp_path):
    # §C.7 parallel self-play CORRECTNESS: a worker playing game-seed S must produce EXACTLY the same examples as
    # sequential self_play_game with random.Random(S) on the same net (search RNG is the passed rng, not torch global).
    from harness.neural import _SELFPLAY_WORKER, _selfplay_worker, save_net
    from harness.registry import resolve_game
    game = Connect4()
    torch.manual_seed(0)
    net = Connect4Net()
    p = str(tmp_path / "net.pt"); save_net(net, p)
    _SELFPLAY_WORKER.clear(); _SELFPLAY_WORKER["game"] = resolve_game("connect4")
    task = (p, 0, None, None, 8, True, 16, 0.1, 0, 2, 8, 12345)  # sims8 gumbel gm16 cs.1 nstep0 open2 temp8 seed12345
    wex = _selfplay_worker(task)
    ag = AlphaZeroAgent(load_net(p), sims=8, gumbel=True, gumbel_m=16, c_scale=0.1)
    sex = self_play_game(game, ag, random.Random(12345), temp_moves=8, target_net=None, n_step=0, opening_plies=2)
    assert len(wex) == len(sex) and len(wex) > 0
    for (xw, piw, vw), (xs, pis, vs) in zip(wex, sex):
        assert piw == pis and vw == vs and torch.equal(xw, xs)


def test_parallel_selfplay_trains_and_returns_valid_buffer():
    # End-to-end: workers>1 fans self-play across processes, trains, returns a valid buffer (spawn path exercised).
    game = Connect4()
    net, hist, buf = train_alphazero(game, iterations=1, selfplay_games=4, sims=6, epochs=1, seed=0,
                                     buffer_cap=5000, selfplay_workers=2, gumbel=True, return_buffer=True)
    assert isinstance(buf, list) and len(buf) > 0
    x, pi, v = buf[0]
    assert tuple(x.shape) == (2, 6, 7) and len(pi) == 7 and -1.0 <= float(v) <= 1.0


def test_parallel_falls_back_when_endgame_on():
    # With the endgame loop armed, parallel MUST fall back to sequential (shared tablebase can't cross processes).
    from harness.tablebase import Tablebase
    game = Connect4()
    tb = Tablebase(cap=200000)
    net, hist, buf = train_alphazero(game, iterations=2, selfplay_games=6, sims=6, epochs=1, seed=3,
                                     endgame_tb=tb, endgame_max_empty=14, selfplay_workers=3, return_buffer=True)
    assert len(tb) > 0  # endgame fired ⇒ it took the sequential endgame path, not the parallel bypass


def test_league_seat_priority_biases_to_p1(monkeypatch):
    # §C.7 #3: league_p1_frac sets P(learner on the won P1 seat). Mutation-guards the seat draw.
    from harness import neural
    from harness.agents import RandomAgent
    game = Connect4()
    seats = []

    def fake_vs(g, learner, opp, seat, rng, record=None):
        seats.append(seat)
        return [(neural.encode(g, g.initial_state(rng)), [1.0] + [0.0] * 6, 1.0)]

    monkeypatch.setattr(neural, "vs_opponent_game", fake_vs)
    neural.train_alphazero(game, iterations=1, selfplay_games=15, sims=6, epochs=1, seed=0,
                           opponent_pool=[lambda: RandomAgent()], pool_frac=1.0, league_p1_frac=1.0)
    assert seats and all(s == 0 for s in seats)
    seats.clear()
    neural.train_alphazero(game, iterations=1, selfplay_games=15, sims=6, epochs=1, seed=0,
                           opponent_pool=[lambda: RandomAgent()], pool_frac=1.0, league_p1_frac=0.0)
    assert seats and all(s == 1 for s in seats)


def test_league_trains_with_anchor_and_frozen_self():
    # The full solver-free league path (real RandomAgent pool + opening anchor + frozen-self) trains cleanly.
    from harness.agents import RandomAgent
    game = Connect4()
    net, hist, buf = train_alphazero(game, iterations=2, selfplay_games=8, sims=6, epochs=1, seed=1,
                                     opponent_pool=[lambda: RandomAgent()], pool_frac=0.5,
                                     league_p1_frac=0.7, opening_anchor_cap=50, league_anchor_frac=0.25,
                                     league_frozen_self=True, return_buffer=True)
    assert isinstance(buf, list) and len(buf) > 0
    assert any(h["vs_pool_games"] > 0 for h in hist)  # league games actually happened


def test_mixed_openings_zero_frac_routes_games_to_the_canonical_line():
    # §C.8 #13 mixed-openings: zero_frac=1.0 ⇒ EVERY game starts canonical (empty-board example present);
    # zero_frac=0.0 with opening_plies>0 ⇒ every game starts off-line (no empty-board example). The mix is the
    # cheap robustness fix: canonical-line sharpness AND off-line coverage from one buffer.
    from harness.neural import train_alphazero

    def run(zero_frac):
        _net, _hist, buf = train_alphazero(
            _game(), iterations=1, selfplay_games=2, sims=6, epochs=1, seed=3, gumbel=True,
            selfplay_opening_plies=6, opening_plies_zero_frac=zero_frac, return_buffer=True,
            net_arch={"channels": 8})
        return buf

    assert any(float(x.abs().sum()) == 0.0 for (x, _pi, _v) in run(1.0))   # canonical: empty board recorded
    assert not any(float(x.abs().sum()) == 0.0 for (x, _pi, _v) in run(0.0))  # off-line only: never the empty board


def test_frontier_order_is_deepest_first_then_most_uncertain():
    # §C.8 #10 (rescoped): net-prioritised frontier — the retrograde invariant (deepest first, children before
    # parents) stays PRIMARY; the net's |value| only breaks ties WITHIN a ply tier (uncertain first), so a tight
    # budget is spent where exact targets teach the most.
    from games.connect4 import Connect4
    from harness.neural import _frontier_order

    game = Connect4()
    rng = random.Random(0)
    s0 = game.initial_state(rng)
    a = game.step(s0, 3, rng)                       # ply 1
    b = game.step(s0, 0, rng)                       # ply 1
    deep = game.step(game.step(s0, 1, rng), 2, rng)  # ply 2 (deepest — must come first regardless of net)
    vals = {id(a): 0.9, id(b): 0.1, id(deep): 0.95}
    order = _frontier_order(game, [a, deep, b], lambda states: [vals[id(s)] for s in states])
    assert order[0] is deep                          # depth beats uncertainty
    assert order[1] is b and order[2] is a           # same ply: |0.1| < |0.9| ⇒ b (uncertain) first
    plain = _frontier_order(game, [a, deep, b], None)  # no net ⇒ legacy deepest-first ordering only
    assert plain[0] is deep


def test_self_play_record_aux_targets_are_mover_relative_and_reply_correct():
    # §C.8 #4: with record_aux, each example carries (own, reply) — own = the FINAL board from THIS position's
    # mover perspective (so consecutive examples are exact negations), reply = the next action played (-1 last).
    game = _game()
    torch.manual_seed(0)
    ex = self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=8), random.Random(4), record_aux=True)
    assert all(len(e) == 5 for e in ex)
    for i, (_x, _pi, _v, own, reply) in enumerate(ex):
        assert own.shape == (42,) and set(own.tolist()) <= {-1.0, 0.0, 1.0}
        if i < len(ex) - 1:
            assert reply in range(7)
            assert torch.equal(own, -ex[i + 1][3])  # same final board, opposite mover perspective
        else:
            assert reply == -1  # the last position has no opponent reply
    pieces = int(ex[0][3].abs().sum())
    assert pieces >= len(ex)  # final board holds at least one piece per recorded ply


def test_augment_mirrors_aux_ownership_and_reply():
    g = Connect4()
    x = torch.zeros(2, 6, 7)
    own = torch.zeros(42)
    own[0] = 1.0  # a piece at row 0, column 0 of the final board
    aug = augment_examples([(x, [1.0, 0, 0, 0, 0, 0, 0], 1.0, own, 0)], g.symmetries())
    _mx, mpi, _mv, mown, mreply = aug[1]
    assert mpi == [0, 0, 0, 0, 0, 0, 1.0]
    assert float(mown[6]) == 1.0 and float(mown[0]) == 0.0  # ownership mirrored to column 6
    assert mreply == 6  # the reply column mirrors with the board
    aug2 = augment_examples([(x, [0.0] * 7, 0.0, own, -1)], g.symmetries())
    assert aug2[1][4] == -1  # no-reply marker survives augmentation


def test_train_net_learns_through_aux_heads_on_mixed_data():
    # Aux losses apply ONLY to examples that carry targets; plain 3-tuples (league/distill) train alongside them.
    game = _game()
    torch.manual_seed(0)
    aux_data, plain = [], []
    for i in range(3):
        aux_data += self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=8), random.Random(i), record_aux=True)
        plain += self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=8), random.Random(100 + i))
    net = Connect4Net(channels=16, blocks=1, residual=True, batchnorm=False, head_hidden=8, aux_heads=True)
    torch.manual_seed(0)
    first = train_net(net, aux_data + plain, epochs=1, batch_size=32, lr=1e-2, device="cpu")
    last = train_net(net, aux_data + plain, epochs=8, batch_size=32, lr=1e-2, device="cpu")
    assert last < first  # the mixed batch trains end-to-end through policy+value+aux
    g = net.own_head.weight.grad if hasattr(net.own_head, "weight") else None  # aux head actually received gradient
    # (gradient presence is asserted structurally: training moved the aux head's parameters)
    before = [p.clone() for p in net.own_head.parameters()]
    train_net(net, aux_data, epochs=1, batch_size=32, lr=1e-2, device="cpu")
    assert any(not torch.equal(b, p) for b, p in zip(before, net.own_head.parameters()))


def test_train_alphazero_with_aux_heads_end_to_end():
    # §C.8 #4 plumbing: an aux-headed arch auto-records targets in self-play, augments them through the mirror,
    # and trains through the aux losses — one iteration end-to-end, no extra knobs.
    net, hist = train_alphazero(
        _game(), iterations=1, selfplay_games=2, sims=6, epochs=1, seed=1, gumbel=True,
        net_arch={"channels": 8, "blocks": 1, "residual": True, "batchnorm": False,
                  "head_hidden": 8, "aux_heads": True})
    assert net.arch["aux_heads"] is True and len(hist) == 1


def test_self_play_forced_opening_replays_the_scripted_prefix():
    # §C.8 #5: a refuted opening is FORCE-replayed (scripted, unrecorded — no fake policy targets for moves the
    # search didn't choose); recording starts at the position the refutation reached.
    game = _game()
    torch.manual_seed(0)
    ex = self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=6), random.Random(2),
                        return_states=True, forced_opening=[3, 3, 3])
    first_state = ex[0][0]
    assert sum(1 for v in first_state.board if v != 0) == 3  # recording begins AFTER the 3-ply prefix
    cols = [i % 7 for i, v in enumerate(first_state.board) if v != 0]
    assert cols == [3, 3, 3]  # and the prefix is exactly the scripted line


def test_vs_opponent_game_record_captures_actions_and_result():
    from harness.agents import RandomAgent

    game = _game()
    torch.manual_seed(0)
    rec = {}
    ex = vs_opponent_game(game, AlphaZeroAgent(Connect4Net(), sims=6), RandomAgent(), 0,
                          random.Random(5), record=rec)
    assert ex and rec["actions"] and all(a in range(7) for a in rec["actions"])
    assert rec["learner_return"] in (-1.0, 0.0, 1.0)
    assert len(rec["actions"]) >= len(ex)  # every recorded learner position came from a played move


def test_train_alphazero_refutation_replay_stocks_and_replays_nogoods():
    # Integration: a stocked store + refutation_frac=1.0 ⇒ self-play games replay the refuted prefix; the buffer
    # then contains positions FROM that line (the nogood is being trained against, not ignored).
    from harness.refutation import RefutationStore

    game = _game()
    store = RefutationStore(cap=8)
    store.add((3, 3, 3))
    _net, _hist, buf = train_alphazero(
        game, iterations=1, selfplay_games=2, sims=6, epochs=1, seed=6, gumbel=True,
        net_arch={"channels": 8}, refutation_frac=1.0, refutation_store=store, return_buffer=True)
    target = None
    rng = random.Random(0)
    s = game.initial_state(rng)
    for a in (3, 3, 3):
        s = game.step(s, a, rng)
    target = encode(game, s)
    assert any(torch.equal(x, target) for (x, _pi, _v) in buf)  # the refuted line's position is in the buffer


def test_resolve_refutation_uses_the_actual_result_not_the_training_target():
    # Review-confirmed HIGH bug guard: with n-step/exact-target training, the first example's VALUE TARGET is a
    # bootstrapped estimate — retirement must track the game's ACTUAL per-seat returns instead.
    from harness.neural import _nogood_prefix, _resolve_refutation
    from harness.refutation import RefutationStore

    store = RefutationStore(cap=4, retire_after=1)
    store.add((3, 3))
    _resolve_refutation(store, (3, 3), {"returns": [-1.0, 1.0]})  # P1 actually LOST the replay
    assert len(store) == 1  # loss ⇒ the nogood stays, whatever any value target claimed
    _resolve_refutation(store, (3, 3), {"returns": [1.0, -1.0]})  # P1 actually won
    assert len(store) == 0  # survival ⇒ retire (retire_after=1)
    store.add((0,))
    _resolve_refutation(store, (0,), None)  # no record (defensive) → no state change, no crash
    _resolve_refutation(store, (0,), {})
    assert len(store) == 1


def test_nogood_prefix_never_stores_the_full_game():
    # Review-confirmed: a full-game prefix replays to a terminal state with ZERO examples and could never retire.
    from harness.neural import _nogood_prefix

    assert _nogood_prefix([0, 3, 1, 3, 0, 3, 1, 3], plies=8) == (0, 3, 1, 3, 0, 3, 1)  # final move excluded
    assert _nogood_prefix([0, 3, 1, 3, 0, 3, 1, 3], plies=4) == (0, 3, 1, 3)  # cap still applies
    assert _nogood_prefix([5], plies=6) == ()  # a 1-move game stores nothing (add() drops empty prefixes)


def test_train_net_can_persist_optimizer_state_across_calls():
    # §C.9 BUG: train_net constructed a fresh Adam every call, so moment estimates reset on EVERY iteration
    # (~200x in a long run) — the optimizer never accumulated. An opt_state dict now carries Adam across calls.
    game = _game()
    torch.manual_seed(0)
    data = self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=8), random.Random(0))
    net = Connect4Net(channels=8)
    state = {}
    train_net(net, data, epochs=1, batch_size=32, lr=1e-3, device="cpu", opt_state=state)
    assert "opt" in state
    opt = state["opt"]
    assert opt.state, "Adam moments must exist after a step"
    train_net(net, data, epochs=1, batch_size=32, lr=1e-3, device="cpu", opt_state=state)
    assert state["opt"] is opt  # SAME optimizer reused, moments carried forward (not reset)
    steps = [s["step"] for s in opt.state.values()]
    assert steps and all(float(s) >= 2 for s in steps)  # step counter advanced across calls


def test_train_net_without_opt_state_is_unchanged():
    # Default path stays byte-identical for every existing caller/run.
    game = _game()
    torch.manual_seed(0)
    data = self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=8), random.Random(1))
    a, b = Connect4Net(channels=8), Connect4Net(channels=8)
    b.load_state_dict(a.state_dict())
    torch.manual_seed(7); la = train_net(a, data, epochs=2, batch_size=16, lr=1e-3, device="cpu")
    torch.manual_seed(7); lb = train_net(b, data, epochs=2, batch_size=16, lr=1e-3, device="cpu", opt_state=None)
    assert la == lb
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_train_net_returns_an_epoch_mean_not_one_minibatch():
    # §C.10 BUILD #3: `final_loss` was `float(loss.detach())` of the LAST (possibly ragged) mini-batch — a
    # single-sample statistic with SD ~0.23 that we then read architectural meaning into. It must be the mean
    # over the final epoch's batches, which is stable enough to compare.
    game = _game()
    torch.manual_seed(0)
    data = []
    for i in range(4):
        data += self_play_game(game, AlphaZeroAgent(Connect4Net(), sims=8), random.Random(i))
    losses = []
    for _ in range(6):
        net = Connect4Net(channels=8)
        torch.manual_seed(0)
        losses.append(train_net(net, data, epochs=1, batch_size=8, lr=1e-3, device="cpu"))
    spread = max(losses) - min(losses)
    assert spread < 0.15, f"epoch-mean loss should be stable across identical runs, spread={spread}"
