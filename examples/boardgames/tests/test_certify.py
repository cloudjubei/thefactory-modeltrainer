"""Direct tests for harness/certify.py — certifying that one player's moves, from a root, always KEEP the position's
exact value against every reply the other side can make. For Connect-4 from the empty board this is §C.49 P-START:
the first player keeps the proven win, so it wins."""
from __future__ import annotations

import random

import pytest

from harness.certify import certify


@pytest.fixture(scope="module")
def ttt():
    from games.tictactoe import TicTacToe

    return TicTacToe()


def _optimal(game):
    from harness.coverage import optimal_actions

    return lambda states: [min(optimal_actions(game, s)) for s in states]


def _value(game):
    return lambda s: game.position_value(s)


def _reference_tree(game, root, player, choose):
    """Every distinct position the walk must visit, by brute force: one move at the player's positions, every move
    at the other side's, stopping at terminals."""
    seen, stack = set(), [root]
    while stack:
        s = stack.pop()
        k = game.state_key(s)
        if k in seen or game.is_terminal(s):
            continue
        seen.add(k)
        moves = [choose([s])[0]] if game.current_player(s) == player else game.legal_actions(s)
        stack.extend(game.step(s, a, random.Random(0)) for a in moves)
    return seen


@pytest.mark.parametrize("player", [0, 1])
def test_an_optimal_player_is_certified_from_the_empty_board_over_every_reply(ttt, player):
    root = ttt.initial_state(random.Random(0))
    r = certify(ttt, root, player, _optimal(ttt), _value(ttt))
    assert r["certified"] and r["complete"] and r["failures"] == 0 and r["root_value"] == 0
    tree = _reference_tree(ttt, root, player, _optimal(ttt))
    assert r["nodes"]["player"] + r["nodes"]["opponent"] == len(tree)
    assert sum(sum(v) for v in r["by_ply"].values()) == len(tree)


def _planted(game, player, root):
    """The optimal chooser, except at ONE reachable position of `player` where it plays a value-losing move."""
    from harness.coverage import optimal_actions

    best = _optimal(game)
    for key in sorted(_reference_tree(game, root, player, best)):
        s = _state_for(game, root, key, player, best)
        if game.current_player(s) != player:
            continue
        bad = [a for a in game.legal_actions(s) if a not in optimal_actions(game, s)]
        if bad:
            return key, bad[0], (lambda states: [bad[0] if game.state_key(x) == key else best([x])[0] for x in states])
    raise AssertionError("no position with a losing move")


def _state_for(game, root, key, player, choose):
    stack = [root]
    while stack:
        s = stack.pop()
        if game.state_key(s) == key:
            return s
        if game.is_terminal(s):
            continue
        moves = [choose([s])[0]] if game.current_player(s) == player else game.legal_actions(s)
        stack.extend(game.step(s, a, random.Random(0)) for a in moves)
    raise KeyError(key)


@pytest.mark.parametrize("player", [0, 1])
def test_one_planted_value_losing_move_is_caught_and_named(ttt, player):
    root = ttt.initial_state(random.Random(0))
    key, action, chooser = _planted(ttt, player, root)
    r = certify(ttt, root, player, chooser, _value(ttt))
    assert not r["certified"] and r["complete"] and r["failures"] == 1
    ex = r["failure_examples"][0]
    assert ex["key"] == key and ex["action"] == action and ex["kept"] < ex["value"]


def test_a_walk_cut_short_by_its_node_budget_is_incomplete_and_never_certified(ttt):
    root = ttt.initial_state(random.Random(0))
    r = certify(ttt, root, 0, _optimal(ttt), _value(ttt), max_nodes=50)
    assert not r["complete"] and not r["certified"] and r["failures"] == 0
    assert r["nodes"]["player"] + r["nodes"]["opponent"] <= 50


def test_the_chooser_is_asked_once_per_ply_for_the_whole_ply(ttt):
    root = ttt.initial_state(random.Random(0))
    batches = []
    best = _optimal(ttt)

    def chooser(states):
        batches.append(len(states))
        return best(states)
    r = certify(ttt, root, 1, chooser, _value(ttt))
    assert len(batches) == sum(1 for p, (mine, _theirs) in r["by_ply"].items() if mine) and max(batches) > 1


def test_an_illegal_move_is_a_failure_not_a_crash(ttt):
    root = ttt.initial_state(random.Random(0))
    r = certify(ttt, root, 0, lambda states: [99 for _ in states], _value(ttt))
    assert not r["certified"] and r["failures"] == 1 and r["failure_examples"][0]["action"] == 99


def _won_connect4_position(seed):
    from games.connect4 import Connect4
    from harness.solver import move_values

    g = Connect4()
    r = random.Random(seed)
    while True:
        s = g.initial_state(r)
        for _ in range(30):
            if g.is_terminal(s):
                break
            s = g.step(s, r.choice(g.legal_actions(s)), r)
        if g.is_terminal(s) or any(g.is_terminal(g.step(s, a, r)) for a in g.legal_actions(s)):
            continue
        vals = move_values(s)
        if max(vals.values()) == 1 and any(v < 1 for v in vals.values()):
            return g, s


def test_a_won_connect4_position_is_certified_for_the_solver_and_a_planted_loser_is_caught():
    from harness.solver import move_values

    g, root = _won_connect4_position(3)
    player = g.current_player(root)

    def solver_move(states):
        return [max(sorted(move_values(s).items()), key=lambda kv: kv[1])[0] for s in states]

    def value(s):
        return max(move_values(s).values())
    r = certify(g, root, player, solver_move, value)
    assert r["certified"] and r["root_value"] == 1 and r["nodes"]["opponent"] > 0
    bad = next(a for a, v in move_values(root).items() if v < 1)
    root_key = g.state_key(root)
    r = certify(g, root, player, lambda states: [bad if g.state_key(s) == root_key else solver_move([s])[0]
                                                 for s in states], value)
    assert not r["certified"] and r["failures"] == 1 and r["failure_examples"][0]["action"] == bad


def test_the_walk_does_not_follow_a_failing_move(ttt):
    root = ttt.initial_state(random.Random(0))
    key, action, chooser = _planted(ttt, 0, root)
    r = certify(ttt, root, 0, chooser, _value(ttt))
    seen, stack = set(), [root]
    while stack:
        s = stack.pop()
        k = ttt.state_key(s)
        if k in seen or ttt.is_terminal(s):
            continue
        seen.add(k)
        if ttt.current_player(s) == 0:
            moves = [] if k == key else chooser([s])
        else:
            moves = ttt.legal_actions(s)
        stack.extend(ttt.step(s, a, random.Random(0)) for a in moves)
    assert r["nodes"]["player"] + r["nodes"]["opponent"] == len(seen)


def test_a_move_onto_an_occupied_cell_is_a_failure_not_a_crash(ttt):
    root = ttt.step(ttt.initial_state(random.Random(0)), 4, random.Random(0))
    r = certify(ttt, root, 1, lambda states: [4 for _ in states], _value(ttt))
    assert not r["certified"] and r["failures"] == 1 and r["failure_examples"][0]["action"] == 4
