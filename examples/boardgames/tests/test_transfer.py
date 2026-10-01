"""Direct tests for harness/transfer.py — the §C.47 probes that place a net's errors relative to its own training
data without enumerating the state space. Tic-tac-toe stands in for Connect-4 because its answers can be checked
exhaustively; nothing under test knows which game it is given."""
from __future__ import annotations

import random

import pytest

from games.tictactoe import TicTacToe
from harness.agents import ExactOptimalAgent
from harness.coverage import _key, move_values, optimal_actions, reachable_states
from harness.transfer import (FURTHER, ONE_MOVE, TWO_MOVES, VISITED, distance_class, first_error, neighbourhood,
                              record_selfplay_states)

G = TicTacToe()
ARCH = {"channels": 16, "blocks": 1, "head_hidden": 8, "residual": True}


def _train(seed=3, **knobs):
    from harness.neural import train_alphazero
    from harness.targets import _weights_sha

    net, history = train_alphazero(G, iterations=2, selfplay_games=3, sims=8, epochs=1, channels=16, net_arch=ARCH,
                                   augment=True, gumbel=True, seed=seed, selfplay_opening_plies=1,
                                   opening_plies_zero_frac=0.5, **knobs)
    return _weights_sha(net), history


@pytest.mark.parametrize("knobs", [{}, {"reanalyze_frac": 1.0, "reanalyze_sims": 4}], ids=["pure", "reanalyze"])
def test_a_recorded_run_trains_bit_identically_and_records_every_game_of_every_iteration(knobs):
    plain, _h = _train(**knobs)
    with record_selfplay_states(G) as log:
        recorded, _h = _train(**knobs)
    assert recorded == plain
    assert [g["pass"] for g in log["games"]] == [0, 0, 0, 1, 1, 1]
    assert log["passes"] == 2
    for g in log["games"]:
        assert g["states"] and not any(G.is_terminal(s) for s in g["states"])


def test_the_recorder_hands_the_caller_exactly_the_shape_it_asked_for():
    import torch

    import harness.neural as neural
    from harness.neural import AlphaZeroAgent, Connect4Net, arch_for_game

    torch.manual_seed(0)
    net = Connect4Net(**arch_for_game(ARCH, G))

    def play(return_states):
        return neural.self_play_game(G, AlphaZeroAgent(net, sims=4, gumbel=True), random.Random(5),
                                     return_states=return_states, opening_plies=1)
    for asked in (False, True):
        plain = play(asked)
        with record_selfplay_states(G) as log:
            seen = play(asked)
        assert len(seen) == len(plain) and all(len(a) == len(b) for a, b in zip(seen, plain))
        for a, b in zip(seen, plain):
            assert all(torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y for x, y in zip(a, b))
        assert len(log["games"]) == 1 and len(log["games"][0]["states"]) == len(plain)


def test_the_recorder_puts_back_what_it_wrapped_however_the_block_exits():
    import harness.neural as neural

    before = (neural.self_play_game, neural.train_net)
    with pytest.raises(RuntimeError):
        with record_selfplay_states(G):
            assert neural.self_play_game is not before[0] and neural.train_net is not before[1]
            raise RuntimeError("boom")
    assert (neural.self_play_game, neural.train_net) == before


def _canonical_at(ply):
    states, _ = reachable_states(G, exact=True, symmetry=True)
    return {_key(G, s) for s in states if G.ply(s) == ply and not G.is_terminal(s)}


@pytest.mark.parametrize("ply", [0, 1, 2])
def test_the_neighbourhood_is_the_visited_keys_then_one_ply_then_two_ply_each_excluding_what_is_nearer(ply):
    states, _ = reachable_states(G, exact=True, symmetry=True)
    seeds = [s for s in states if G.ply(s) == ply][:2]
    hood = neighbourhood(G, seeds)
    assert hood["visited"] == {_key(G, s) for s in seeds}
    one = {_key(G, G.step(s, a)) for s in seeds for a in G.legal_actions(s)} - hood["visited"]
    assert hood["one_move_off"] == one
    assert hood["two_moves_off"] <= _canonical_at(ply + 2) and hood["two_moves_off"].isdisjoint(one)
    assert hood["one_move_off"] <= _canonical_at(ply + 1)


def test_the_neighbourhood_of_the_empty_board_is_the_three_openings_then_the_twelve_replies():
    hood = neighbourhood(G, [G.initial_state(random.Random(0))])
    assert len(hood["visited"]) == 1 and hood["one_move_off"] == _canonical_at(1) and len(hood["one_move_off"]) == 3
    assert hood["two_moves_off"] == _canonical_at(2) and len(hood["two_moves_off"]) == 12


def test_a_visited_LINE_is_excluded_from_its_own_rings_even_where_the_rings_would_reach_it():
    """Self-play records whole games, so the visited set holds positions one and two plies apart: a ring that did
    not exclude the nearer rings would count a trained position as merely reachable."""
    line = [G.initial_state(random.Random(0))]
    for a in (4, 0, 8):
        line.append(G.step(line[-1], a))
    hood = neighbourhood(G, line)
    children = {_key(G, G.step(s, a)) for s in line for a in G.legal_actions(s)}
    assert children & hood["visited"], "the fixture must let a ring reach a visited position"
    assert hood["one_move_off"].isdisjoint(hood["visited"])
    grandchildren = {_key(G, G.step(G.step(s, a), b)) for s in line for a in G.legal_actions(s)
                     for b in G.legal_actions(G.step(s, a)) if not G.is_terminal(G.step(s, a))}
    assert grandchildren & hood["one_move_off"], "the fixture must let the two-ply ring reach the one-ply ring"
    assert hood["two_moves_off"].isdisjoint(hood["one_move_off"] | hood["visited"])


def test_terminal_children_are_not_part_of_the_neighbourhood():
    states, _ = reachable_states(G, exact=True, symmetry=True)
    near_end = [s for s in states if not G.is_terminal(s)
                and any(G.is_terminal(G.step(s, a)) for a in G.legal_actions(s))][:5]
    hood = neighbourhood(G, near_end)
    terminal = {_key(G, G.step(s, a)) for s in near_end for a in G.legal_actions(s) if G.is_terminal(G.step(s, a))}
    assert terminal and terminal.isdisjoint(hood["one_move_off"])


def test_distance_class_names_every_ring():
    root = G.initial_state(random.Random(0))
    hood = neighbourhood(G, [root])
    one = G.step(root, 4)
    two = G.step(one, 0)
    three = G.step(two, 8)
    assert [distance_class(G, s, hood) for s in (root, one, two, three)] == [VISITED, ONE_MOVE, TWO_MOVES, FURTHER]


def _proven_wins():
    states, _ = reachable_states(G, exact=True, symmetry=True)
    return [s for s in states if not G.is_terminal(s) and max(move_values(G, s).values()) > 0]


def test_an_exact_policy_converts_every_proven_win_against_an_exact_defender():
    roots = _proven_wins()
    assert len(roots) > 50
    for root in roots:
        r = first_error(G, lambda s: min(optimal_actions(G, s)), root, ExactOptimalAgent())
        assert r["converted"] and r["plies"] >= 1 and "state" not in r


def test_the_first_error_is_the_first_position_where_the_move_gives_the_win_away():
    roots = [r for r in _proven_wins() if len(G.legal_actions(r)) >= 4]
    checked = 0
    for root in roots[:40]:
        vals = move_values(G, root)
        losing = [a for a, v in vals.items() if v <= 0]
        if not losing:
            continue
        r = first_error(G, lambda s: losing[0] if s == root else min(optimal_actions(G, s)), root, ExactOptimalAgent())
        assert not r["converted"] and r["state"] == root and r["move"] == losing[0] and r["plies"] == 0
        assert r["winning"] == sorted(a for a, v in vals.items() if v > 0)
        checked += 1
    assert checked >= 10


def test_an_error_after_correct_moves_is_found_at_its_own_ply():
    for root in _proven_wins():
        mover = G.current_player(root)
        s, d = root, ExactOptimalAgent()
        s = G.step(s, min(optimal_actions(G, s)))
        if G.is_terminal(s):
            continue
        s = G.step(s, d.act(G, s, random.Random(0)))
        if G.is_terminal(s) or G.current_player(s) != mover:
            continue
        bad = [a for a, v in move_values(G, s).items() if v <= 0]
        if not bad:
            continue
        target = s

        def policy(state):
            return bad[0] if state == target else min(optimal_actions(G, state))
        r = first_error(G, policy, root, ExactOptimalAgent())
        assert not r["converted"] and r["state"] == target and r["plies"] == 2
        return
    pytest.fail("no root with a two-ply error line")


def test_a_root_without_a_proven_win_is_refused():
    root = G.initial_state(random.Random(0))
    with pytest.raises(ValueError, match="no proven win"):
        first_error(G, lambda s: min(optimal_actions(G, s)), root, ExactOptimalAgent())


def test_a_value_oracle_that_disagrees_with_the_game_is_refused():
    root = next(r for r in _proven_wins() if move_values(G, r)[max(G.legal_actions(r))] <= 0)

    def lying(s):
        return {a: 1 for a in G.legal_actions(s)}
    with pytest.raises(ValueError, match="disagree"):
        first_error(G, lambda s: max(G.legal_actions(s)), root, ExactOptimalAgent(), values_fn=lying)


class _FixedNet:
    """A stand-in net whose logits are fixed per row, with a train/eval mode flag to prove it is restored."""

    def __init__(self, logits):
        import torch
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def __call__(self, x):
        return self.logits[: len(x)], None


def test_policy_accuracy_masks_illegal_moves_and_puts_the_net_back_in_its_mode():
    import torch

    from harness.transfer import policy_accuracy
    probe = {"x": torch.zeros(3, 2), "legal": torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.bool),
             "optimal": [{1}, {2}, {1}]}
    net = _FixedNet([[0.1, 0.5, 9.0], [5.0, 0.2, 0.3], [0.0, 3.0, 1.0]])
    assert policy_accuracy(net, probe) == pytest.approx(2 / 3)
    assert net.training is True
    net.train(False)
    policy_accuracy(net, probe)
    assert net.training is False


def test_build_probe_records_each_position_s_legal_moves_and_optimal_set():
    from harness.neural import encode
    from harness.transfer import build_probe
    states = [s for s in reachable_states(G, exact=True, symmetry=True)[0] if not G.is_terminal(s)][:25]
    probe = build_probe(G, states, encode)
    assert probe["x"].shape[0] == 25 and probe["optimal"] == [optimal_actions(G, s) for s in states]
    assert [sorted(probe["legal"][i].nonzero().flatten().tolist()) for i in range(25)] == \
        [sorted(G.legal_actions(s)) for s in states]


def test_a_probed_recorded_run_is_still_bit_identical_and_probes_once_per_pass():
    from harness.neural import encode
    from harness.transfer import build_probe
    states = [s for s in reachable_states(G, exact=True, symmetry=True)[0] if not G.is_terminal(s)][:40]
    plain, _h = _train(reanalyze_frac=1.0, reanalyze_sims=4)
    with record_selfplay_states(G, probe=build_probe(G, states, encode)) as log:
        recorded, _h = _train(reanalyze_frac=1.0, reanalyze_sims=4)
    assert recorded == plain and len(log["probe"]) == 2 and all(0 <= a <= 1 for a in log["probe"])


def test_an_observer_is_handed_the_net_after_every_pass_and_the_run_stays_bit_identical():
    from harness.targets import _weights_sha

    plain, _h = _train(reanalyze_frac=1.0, reanalyze_sims=4)
    seen = []

    def observe(net):
        seen.append(_weights_sha(net))
        return len(seen) * 10
    with record_selfplay_states(G, on_pass=observe) as log:
        recorded, _h = _train(reanalyze_frac=1.0, reanalyze_sims=4)
    assert recorded == plain and log["on_pass"] == [10, 20]
    assert len(set(seen)) == 2 and seen[-1] == recorded
    with record_selfplay_states(G) as log:
        _train()
    assert log["on_pass"] == []


def test_forbid_solver_makes_every_exact_solve_raise_and_restores_it():
    import harness.solver as solver
    from harness.registry import resolve_game
    from harness.transfer import forbid_solver
    c4 = resolve_game("connect4")
    s = c4.initial_state(random.Random(0))
    for a in (3, 3, 2, 4, 2, 4, 1, 5, 1, 5, 0, 6, 0, 6, 3, 3, 2, 2, 4, 4):
        s = c4.step(s, a)
    before = solver.move_values(s)
    with pytest.raises(RuntimeError, match="solver-free"):
        with forbid_solver():
            try:
                solver.move_values(s)
            except RuntimeError:
                c4.position_value(s)
    assert solver.move_values(s) == before


@pytest.mark.parametrize("entry", ["move_values", "position_value", "solve_position"])
def test_forbid_solver_also_refuses_every_native_solver_entry_and_restores_it(entry):
    from harness import native_solver
    from harness.registry import resolve_game
    from harness.transfer import forbid_solver
    c4 = resolve_game("connect4")
    s = c4.initial_state(random.Random(0))
    for a in (3, 3, 2, 4, 2, 4, 1, 5, 1, 5, 0, 6, 0, 6, 3, 3, 2, 2, 4, 4):
        s = c4.step(s, a)
    before = getattr(native_solver, entry)(s)
    with pytest.raises(RuntimeError, match="solver-free"):
        with forbid_solver():
            getattr(native_solver, entry)(s)
    assert getattr(native_solver, entry)(s) == before


def test_the_connect4_stage0_recipe_trains_with_the_solver_forbidden():
    from harness.neural import train_alphazero
    from harness.registry import resolve_game
    from harness.transfer import forbid_solver
    c4 = resolve_game("connect4")
    with forbid_solver():
        net, history = train_alphazero(c4, iterations=2, selfplay_games=2, sims=4, epochs=1, channels=8,
                                       net_arch={"channels": 8, "blocks": 1, "head_hidden": 8, "residual": True},
                                       augment=True, gumbel=True, seed=1, selfplay_opening_plies=4,
                                       opening_plies_zero_frac=0.3, reanalyze_frac=1.0, reanalyze_sims=4)
    assert len(history) == 2


def test_a_position_is_nontrivial_only_if_failable_without_an_immediate_win():
    from harness.transfer import nontrivial
    states, _ = reachable_states(G, exact=True, symmetry=True)
    seen = {True: 0, "win_in_one": 0, "uniform": 0}
    for s in states:
        if G.is_terminal(s):
            assert not nontrivial(G, s, {})
            continue
        vals = move_values(G, s)
        mover = G.current_player(s)
        win_now = any(G.is_terminal(G.step(s, a)) and G.returns(G.step(s, a))[mover] > 0 for a in G.legal_actions(s))
        uniform = len(set(vals.values())) < 2
        assert nontrivial(G, s, vals) == (not win_now and not uniform)
        seen[True] += nontrivial(G, s, vals)
        seen["win_in_one"] += win_now and not uniform
        seen["uniform"] += uniform
    assert all(v > 10 for v in seen.values())


@pytest.mark.parametrize("line", [(4, 0, 8), (0, 4, 1, 2), (4, 1, 0, 8, 2)])
def test_rings_at_a_ply_are_the_full_neighbourhood_restricted_to_that_ply(line):
    from harness.transfer import rings_at
    states = [G.initial_state(random.Random(0))]
    for a in line:
        states.append(G.step(states[-1], a))
    extra = [s for s in reachable_states(G, exact=True, symmetry=True)[0] if G.ply(s) == 2][:3]
    visited = states + extra
    hood = neighbourhood(G, visited)
    keyed = {_key(G, s): s for s in reachable_states(G, exact=True, symmetry=True)[0]}
    for ply in range(1, 7):
        r = rings_at(G, visited, ply)
        assert set(r["one_move_off"]) == {k for k in hood["one_move_off"] if G.ply(keyed[k]) == ply}
        assert set(r["two_moves_off"]) == {k for k in hood["two_moves_off"] if G.ply(keyed[k]) == ply}
        assert all(G.ply(st) == ply for ring in r.values() for st in ring.values())


def test_the_recorder_logs_every_sibling_set_the_trainer_relabels_with_its_pass():
    from games.tictactoe import TicTacToe
    from harness.neural import train_alphazero
    from harness.transfer import record_selfplay_states

    g = TicTacToe()
    kw = dict(iterations=3, selfplay_games=2, sims=4, epochs=1, net_arch={"channels": 4}, augment=False, seed=3,
              reanalyze_frac=1.0, reanalyze_sims=4, reanalyze_siblings=True)
    with record_selfplay_states(g) as log:
        net, history = train_alphazero(g, **kw)
    assert [row["pass"] for row in log["siblings"]] == [1, 2]
    assert [len(row["states"]) for row in log["siblings"]] == [h["siblings"] for h in history[1:]]
    plain, _ = train_alphazero(g, **kw)
    import torch

    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), plain.state_dict().values()))
