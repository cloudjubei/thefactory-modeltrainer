"""Direct tests for harness.targets — splitting one self-play search into what the PRIOR wanted, what the SEARCH
found (completed Q) and what the TARGET recorded (the completed-Q policy the net is trained to imitate). A stub net
with a fixed, confidently WRONG prior makes each failure class known by construction."""
from __future__ import annotations

import hashlib
import random

import pytest
import torch

import harness.neural as neural
from games.tictactoe import TicTacToe, TTTState
from harness.coverage import optimal_actions, reachable_states
from harness.neural import AlphaZeroAgent, Connect4Net, arch_for_game, encode
from harness.targets import (POLICY_TARGETS, exact_policy_target, record_training_labels, record_training_passes,
                             search_decomposition, target_error)

G = TicTacToe()
# X to move; X wins at once on 2 and forks on 3 (both optimal); 6/7/8 let O win on 3.
WIN = TTTState(board=(1, 1, 0, 0, 2, 2, 0, 0, 0), to_move=0, winner=None, done=False)
# WIN's left-right mirror: the same canonical key, but its optimal moves are {0, 5}, not {2, 3}.
MIRROR = TTTState(board=(0, 1, 1, 2, 2, 0, 0, 0, 0), to_move=0, winner=None, done=False)
STATES, _ = reachable_states(G, exact=True)


class _StubNet(torch.nn.Module):
    """A prior that always puts nearly all its mass on `favourite`, and a value head that says 0 everywhere."""

    num_actions = 9

    def __init__(self, favourite: int, confidence: float = 8.0):
        super().__init__()
        self.favourite, self.confidence = favourite, confidence

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 9)
        logits[:, self.favourite] = self.confidence
        return logits, torch.zeros(x.shape[0], 1)


def _agent(sims, c_scale, favourite=7):
    return lambda: AlphaZeroAgent(_StubNet(favourite), sims=sims, gumbel=True, c_scale=c_scale)


def test_the_decomposition_reports_the_prior_the_search_and_the_target_separately():
    d = search_decomposition(G, _agent(64, 1.0), WIN)
    assert d["prior"] == 7
    assert d["q_best"] == [2]
    assert d["label"] in optimal_actions(G, WIN)
    assert sum(d["visits"].values()) == 64


def test_a_search_that_FOUND_the_win_but_a_target_that_trusts_the_prior_is_PRIOR_ANCHOR():
    d = search_decomposition(G, _agent(64, 0.0), WIN)
    assert d["q_best"] == [2] and d["label"] == 7
    assert target_error(d, optimal_actions(G, WIN)) == "prior_anchor"


def test_a_search_too_shallow_to_see_the_win_is_a_SEARCH_MISS():
    d = search_decomposition(G, _agent(1, 0.1), WIN)
    assert d["label"] == 7
    assert not set(d["q_best"]) <= optimal_actions(G, WIN)
    assert target_error(d, optimal_actions(G, WIN)) == "search_miss"


def test_a_correct_target_is_LABEL_OK_whatever_the_prior_wanted():
    d = search_decomposition(G, _agent(64, 1.0), WIN)
    assert target_error(d, optimal_actions(G, WIN)) == "label_ok"


def test_a_TIE_for_the_best_Q_counts_as_found_only_if_EVERY_tied_move_is_optimal():
    tie = {"label": 7, "q_best": [2, 7]}
    assert target_error(tie, {2, 3}) == "search_miss"
    both = {"label": 7, "q_best": [2, 3]}
    assert target_error(both, {2, 3}) == "prior_anchor"


def test_every_call_searches_from_a_FRESH_agent_so_repeated_calls_agree():
    first = search_decomposition(G, _agent(16, 0.1), WIN)
    again = search_decomposition(G, _agent(16, 0.1), WIN)
    assert first["visits"] == again["visits"] and first["label"] == again["label"]


def test_temperature_turns_on_the_self_play_gumbel_noise_and_the_seed_fixes_it():
    a = search_decomposition(G, _agent(16, 0.1, favourite=0), G.initial_state(), seed=3, temperature=1.0)
    b = search_decomposition(G, _agent(16, 0.1, favourite=0), G.initial_state(), seed=3, temperature=1.0)
    quiet = search_decomposition(G, _agent(16, 0.1, favourite=0), G.initial_state(), seed=3, temperature=0.0)
    assert a == b
    noisy = [search_decomposition(G, _agent(16, 0.1, favourite=0), G.initial_state(), seed=s, temperature=1.0)
             for s in range(8)]
    assert any(n["visits"] != quiet["visits"] for n in noisy)


def _tiny(iterations=2, **kw):
    from harness.neural import train_alphazero
    return train_alphazero(G, iterations=iterations, selfplay_games=3, sims=8, epochs=1, channels=8,
                           net_arch={"channels": 8}, gumbel=True, seed=5, **kw)[0]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_recording_the_trained_on_labels_does_NOT_change_what_is_trained():
    plain = _tiny()
    with record_training_labels(G, {G.canonical_key(G.initial_state())}, encode) as log:
        watched = _tiny()
    assert _same(plain, watched)
    assert len(log) == 2


def test_the_recorder_reports_each_training_pass_s_dose_at_the_target_states():
    target = {G.canonical_key(s) for s in STATES if G.ply(s) == 1}
    with record_training_labels(G, target, encode) as log:
        _tiny()
    assert [row["pass"] for row in log] == [1, 2]
    assert all(row["n_target"] > 0 and 0 <= row["argmax_ok"] <= row["n_target"] for row in log)
    assert all(row["examples"] >= row["n_target"] for row in log)
    import json
    for row in log:
        kept = json.loads(json.dumps(row["per_key"]))
        assert kept == [[k, v] for k, v in row["per_key"]] and all(isinstance(k, int) for k, _ in kept)
        assert sum(v["n"] for _, v in row["per_key"]) == row["n_target"]


def test_the_recorder_restores_training_even_when_the_run_raises():
    original = neural.train_net
    try:
        with record_training_labels(G, set(), encode):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert neural.train_net is original


BLOCK = TTTState(board=(1, 1, 0, 0, 2, 0, 0, 0, 0), to_move=1, winner=None, done=False)  # O must block on 2
BLOCK_MIRROR = TTTState(board=(0, 1, 1, 0, 2, 0, 0, 0, 0), to_move=1, winner=None, done=False)  # ...here on 0


class _OneMoveGame:
    """One decision, then the game is over with the mover's return `payoff[action]`. `num_actions` is wider than
    the legal set, so the target's width and its zeros on illegal moves can only come from the game."""

    num_actions = 5

    def __init__(self, payoff: dict):
        self.payoff = payoff

    def is_terminal(self, s):
        return s != "root"

    def current_player(self, s):
        return 0

    def legal_actions(self, s):
        return sorted(self.payoff) if s == "root" else []

    def step(self, s, a):
        return a

    def returns(self, s):
        return [self.payoff[s], -self.payoff[s]]


def test_the_exact_target_is_UNIFORM_over_the_solver_s_optimal_moves_and_zero_elsewhere():
    sizes = set()
    for s in STATES + [WIN, MIRROR, BLOCK, BLOCK_MIRROR]:
        pi, opt = exact_policy_target(G, s), optimal_actions(G, s)
        assert len(pi) == G.num_actions
        assert {a for a, p in enumerate(pi) if p > 0} == opt
        assert all(pi[a] == 1.0 / len(opt) for a in opt)
        assert abs(sum(pi) - 1.0) < 1e-9
        sizes.add(len(opt))
    assert {1, 2, 3, 9} <= sizes
    assert exact_policy_target(G, WIN) == [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert exact_policy_target(G, MIRROR) == [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("payoff, expected", [
    ({0: -1.0, 2: 1.0, 3: 1.0, 4: 0.0}, [0.0, 0.0, 0.5, 0.5, 0.0]),
    ({1: 0.0, 4: 0.0}, [0.0, 0.5, 0.0, 0.0, 0.5]),
    ({0: -1.0, 1: -1.0, 3: 0.0}, [0.0, 0.0, 0.0, 1.0, 0.0]),
])
def test_the_exact_target_is_as_wide_as_the_GAME_s_action_space(payoff, expected):
    assert exact_policy_target(_OneMoveGame(payoff), "root") == expected


@pytest.mark.parametrize("moves", [(0, 3, 1, 4, 2), (0, 1, 2, 4, 3, 5, 7, 6, 8)])
def test_the_exact_target_REFUSES_a_terminal_state(moves):
    s = G.initial_state()
    for a in moves:
        s = G.step(s, a)
    assert G.is_terminal(s)
    with pytest.raises(ValueError):
        exact_policy_target(G, s)
    with pytest.raises(ValueError):
        exact_policy_target(_OneMoveGame({0: 1.0, 1: 0.0}), 1)


def test_a_config_names_the_exact_target_by_STRING():
    assert POLICY_TARGETS == {"exact_uniform_optimal": exact_policy_target}


def _legacy_net(seed: int = 0):
    torch.manual_seed(seed)
    return Connect4Net(**arch_for_game({"channels": 8}, G))


def _sha(net):
    h = hashlib.sha256()
    for t in net.state_dict().values():
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _direct_policy_fails(net, states):
    """The raw-policy failures one state at a time, the way localize_ceiling measured them before §C.46."""
    net.eval()
    out = []
    for s in states:
        with torch.no_grad():
            logits, _v = net(encode(G, s).unsqueeze(0))
        move = max(G.legal_actions(s), key=lambda a: float(logits[0, a]))
        if move not in optimal_actions(G, s):
            out.append(G.canonical_key(s))
    return sorted(out)


def _no_op_train(net, examples, *args, **kwargs):
    return 0.0


def _train_mode_train(net, examples, *args, **kwargs):
    net.train()
    return 0.0


def _eval_mode_train(net, examples, *args, **kwargs):
    net.eval()
    return 0.0


def _pi(top: int, soft: bool) -> list:
    if not soft:
        return [1.0 if a == top else 0.0 for a in range(9)]
    return [0.6 if a == top else 0.05 for a in range(9)]


NAN = float("nan")
# (x, pi, v). A mirror image shares its twin's canonical key but not its optimal set, so every label must be graded
# in its OWN frame: graded in the twin's frame, the MIRROR and BLOCK_MIRROR rows below score differently.
DOSE_EXAMPLES = [
    (encode(G, WIN), _pi(2, soft=False), 1.0),
    (encode(G, MIRROR), _pi(0, soft=True), NAN),
    (encode(G, MIRROR), _pi(5, soft=True), NAN),
    (encode(G, MIRROR), _pi(3, soft=False), NAN),
    (encode(G, G.initial_state()), _pi(4, soft=True), 0.0),
    (encode(G, BLOCK), _pi(6, soft=True), -1.0),
    (encode(G, BLOCK_MIRROR), _pi(0, soft=False), NAN),
    (torch.ones(2, 3, 3), _pi(0, soft=False), 1.0),
]
WIN_KEY, BLOCK_KEY, INITIAL_KEY = G.canonical_key(WIN), G.canonical_key(BLOCK), G.canonical_key(G.initial_state())


def test_the_dose_fixture_grades_a_mirror_image_differently_from_its_canonical_twin():
    assert G.canonical_key(MIRROR) == WIN_KEY and G.canonical_key(BLOCK_MIRROR) == BLOCK_KEY
    assert optimal_actions(G, WIN) == {2, 3} and optimal_actions(G, MIRROR) == {0, 5}
    assert optimal_actions(G, BLOCK) == {2} and optimal_actions(G, BLOCK_MIRROR) == {0}


def test_the_dose_splits_SIBLING_rows_from_SELF_PLAY_rows_by_the_NaN_value_rule(monkeypatch):
    monkeypatch.setattr(neural, "train_net", _no_op_train)
    with record_training_passes(G, STATES, encode) as passes:
        neural.train_net(_legacy_net(), DOSE_EXAMPLES, 1, 64, 1e-3, "cpu")
    expected = {WIN_KEY: [1, 1, 3, 2], INITIAL_KEY: [1, 1, 0, 0], BLOCK_KEY: [1, 0, 1, 1]}
    keys = sorted(G.canonical_key(s) for s in STATES)
    assert passes[0]["dose"] == [[k, *expected.get(k, [0, 0, 0, 0])] for k in keys]


def test_the_dose_covers_exactly_the_keys_of_the_states_it_was_given(monkeypatch):
    monkeypatch.setattr(neural, "train_net", _no_op_train)
    subset = [s for s in STATES if G.canonical_key(s) in {WIN_KEY, BLOCK_KEY, 7, 163}]
    assert len(subset) == 4
    with record_training_passes(G, subset, encode) as passes:
        neural.train_net(_legacy_net(), DOSE_EXAMPLES, 1, 64, 1e-3, "cpu")
        neural.train_net(_legacy_net(), DOSE_EXAMPLES[:2], 1, 64, 1e-3, "cpu")
    assert passes[0]["dose"] == [[7, 0, 0, 0, 0], [163, 0, 0, 0, 0], [BLOCK_KEY, 1, 0, 1, 1], [WIN_KEY, 1, 1, 3, 2]]
    assert passes[1]["dose"] == [[7, 0, 0, 0, 0], [163, 0, 0, 0, 0], [BLOCK_KEY, 0, 0, 0, 0], [WIN_KEY, 1, 1, 1, 1]]
    assert [p["pass"] for p in passes] == [1, 2]


def test_the_dose_is_what_the_pass_was_FED_even_if_the_pass_consumes_its_examples(monkeypatch):
    def consuming_train(net, examples, *args, **kwargs):
        examples.clear()
        return 0.0

    monkeypatch.setattr(neural, "train_net", consuming_train)
    fed = list(DOSE_EXAMPLES)
    with record_training_passes(G, STATES, encode) as passes:
        neural.train_net(_legacy_net(), fed, 1, 64, 1e-3, "cpu")
    assert fed == []
    assert {k: row for k, *row in passes[0]["dose"] if any(row)} == {
        WIN_KEY: [1, 1, 3, 2], INITIAL_KEY: [1, 1, 0, 0], BLOCK_KEY: [1, 0, 1, 1]}


def test_the_solver_is_asked_ONCE_per_raw_position_however_many_passes_see_it(monkeypatch):
    import harness.targets as targets
    asked = []
    solve = targets.optimal_actions

    def counting(game, s):
        asked.append(game.state_key(s))
        return solve(game, s)

    monkeypatch.setattr(targets, "optimal_actions", counting)
    monkeypatch.setattr(neural, "train_net", _no_op_train)
    with record_training_passes(G, STATES, encode):
        for _ in range(3):
            neural.train_net(_legacy_net(), DOSE_EXAMPLES, 1, 64, 1e-3, "cpu")
    assert len(asked) == len(set(asked))
    assert set(asked) == {G.state_key(s) for s in STATES + [MIRROR, BLOCK_MIRROR]}


class _RampNet(torch.nn.Module):
    """FIXED logits, one per action, whatever the board: its masked argmax at a state is its highest-ranked LEGAL
    move, so the right answer is computable by hand. The logits are a Parameter so there are bytes to hash."""

    def __init__(self, logits: list):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1), torch.zeros(x.shape[0], 1)


@pytest.mark.parametrize("logits", [[5, 1, 4, 2, 9, 3, 6, 0, 7], [0, 8, 1, 7, 2, 6, 3, 5, 4]])
@pytest.mark.parametrize("pick", ["all", "early", "every_third"])
def test_policy_fail_keys_MASK_illegal_moves_before_taking_the_argmax(monkeypatch, logits, pick):
    monkeypatch.setattr(neural, "train_net", _no_op_train)
    states = {"all": STATES, "early": [s for s in STATES if G.ply(s) <= 4], "every_third": STATES[::3]}[pick]
    with record_training_passes(G, states, encode) as passes:
        neural.train_net(_RampNet(logits), [], 1, 64, 1e-3, "cpu")
    masked = sorted(G.canonical_key(s) for s in states
                    if max(G.legal_actions(s), key=lambda a: logits[a]) not in optimal_actions(G, s))
    unmasked = sorted(G.canonical_key(s) for s in states
                      if max(range(9), key=lambda a: logits[a]) not in optimal_actions(G, s))
    assert masked != unmasked
    assert passes[0]["policy_fail_keys"] == masked


class _NormNet(torch.nn.Module):
    """Logits through a BatchNorm: a forward in TRAIN mode moves its running statistics — state the next pass
    would train from — and a forward in EVAL mode does not."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(18, 9)
        self.norm = torch.nn.BatchNorm1d(9)

    def forward(self, x):
        return self.norm(self.lin(x.flatten(1))), torch.zeros(x.shape[0], 1)


def test_the_measurement_forward_runs_in_EVAL_mode_and_never_moves_the_net_s_state(monkeypatch):
    monkeypatch.setattr(neural, "train_net", _train_mode_train)
    net = _NormNet()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with record_training_passes(G, STATES, encode) as passes:
        neural.train_net(net, [], 1, 64, 1e-3, "cpu")
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
    assert passes[0]["policy_fail_keys"] == _direct_policy_fails(net, STATES)


@pytest.mark.parametrize("train, mode", [(_train_mode_train, True), (_eval_mode_train, False)])
def test_the_net_is_left_in_the_mode_the_training_pass_left_it_in(monkeypatch, train, mode):
    monkeypatch.setattr(neural, "train_net", train)
    net = _NormNet()
    net.train(not mode)
    with record_training_passes(G, STATES, encode):
        neural.train_net(net, [], 1, 64, 1e-3, "cpu")
    assert net.training is mode


class _BrokenNet(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("forward failed")


def test_the_net_s_mode_is_restored_even_when_the_measurement_forward_raises(monkeypatch):
    monkeypatch.setattr(neural, "train_net", _no_op_train)
    net = _BrokenNet()
    net.train()
    with record_training_passes(G, STATES, encode):
        with pytest.raises(RuntimeError, match="forward failed"):
            neural.train_net(net, [], 1, 64, 1e-3, "cpu")
    assert net.training


def test_the_pass_recorder_passes_every_argument_through_returns_the_result_and_draws_no_rng(monkeypatch):
    seen = []

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return 0.375

    monkeypatch.setattr(neural, "train_net", spy)
    net, box = _legacy_net(), {}
    examples = DOSE_EXAMPLES[:3]
    torch_rng, py_rng = torch.get_rng_state(), random.getstate()
    with record_training_passes(G, STATES, encode) as passes:
        out = neural.train_net(net, examples, 3, 16, 0.5, "cpu", opt_state=box, epoch_examples=7)
    assert out == 0.375
    [(args, kwargs)] = seen
    assert args[0] is net and args[1] is examples and args[2:] == (3, 16, 0.5, "cpu")
    assert kwargs == {"opt_state": box, "epoch_examples": 7} and kwargs["opt_state"] is box
    assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == py_rng
    assert len(passes) == 1


def test_the_pass_recorder_restores_training_even_when_the_run_raises():
    original = neural.train_net
    with pytest.raises(RuntimeError, match="boom"):
        with record_training_passes(G, STATES, encode):
            assert neural.train_net is not original
            raise RuntimeError("boom")
    assert neural.train_net is original


def test_recording_the_passes_does_NOT_change_what_is_trained():
    plain = _tiny()
    with record_training_passes(G, STATES, encode) as passes:
        watched = _tiny()
    assert _same(plain, watched)
    assert [p["pass"] for p in passes] == [1, 2]


def test_both_recorders_NEST_in_either_order_and_each_records_what_it_records_alone():
    target = {INITIAL_KEY, BLOCK_KEY}
    plain = _tiny()
    with record_training_labels(G, target, encode) as labels_alone:
        _tiny()
    with record_training_passes(G, STATES, encode) as passes_alone:
        _tiny()
    original = neural.train_net
    with record_training_labels(G, target, encode) as labels_outer, \
            record_training_passes(G, STATES, encode) as passes_inner:
        inner_last = _tiny()
    assert neural.train_net is original
    with record_training_passes(G, STATES, encode) as passes_outer, \
            record_training_labels(G, target, encode) as labels_inner:
        labels_last = _tiny()
    assert neural.train_net is original
    assert _same(plain, inner_last) and _same(plain, labels_last)
    assert len(labels_alone) == len(passes_alone) == 2
    assert labels_outer == labels_inner == labels_alone
    assert passes_inner == passes_outer == passes_alone
    for p, lab in zip(passes_alone, labels_alone):
        assert sum(row[1] for row in p["dose"]) == lab["examples"]
        assert sum(row[3] for row in p["dose"]) == 0


@pytest.fixture(scope="module")
def recorded_runs():
    with record_training_passes(G, STATES, encode) as one:
        first = _tiny(iterations=1)
    with record_training_passes(G, STATES, encode) as two:
        final = _tiny()
    return one, first, two, final


def test_weights_sha_hashes_the_net_AFTER_each_pass_and_changes_between_passes(recorded_runs):
    one, first, two, final = recorded_runs
    assert len({p["weights_sha"] for p in two}) == 2
    assert two[0]["weights_sha"] == one[0]["weights_sha"] == _sha(first)
    assert two[1]["weights_sha"] == _sha(final)
    assert _sha(_legacy_net(5)) != two[0]["weights_sha"]


def test_policy_fail_keys_are_the_raw_policy_failures_AFTER_each_pass(recorded_runs):
    one, first, two, final = recorded_runs
    assert two[1]["policy_fail_keys"] == _direct_policy_fails(final, STATES)
    assert one[0]["policy_fail_keys"] == two[0]["policy_fail_keys"] == _direct_policy_fails(first, STATES)
    assert two[0]["policy_fail_keys"] != two[1]["policy_fail_keys"]
    assert 0 < len(two[1]["policy_fail_keys"]) < len(STATES)
