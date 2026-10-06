"""Direct tests for harness/strategy_size.py — how small a certified strategy for one side can be: the game graph's
size, the canonical strategy (the first value-keeping move in action order, every reply), and the strategy whose
value-keeping choices minimise the decisions in its tree. Exact values come from the game's own solver."""
from __future__ import annotations

import pytest

from games.kalah import Kalah
from games.tictactoe import TicTacToe, TTTState
from harness.certify import certify
from harness.strategy_size import canonical_strategy, decisions, game_graph_size, min_tree_strategy


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


@pytest.mark.parametrize("shape,size", [((1, 1), 2), ((1, 4), 3), ((2, 1), 8)])
def test_the_game_graph_counts_every_reachable_position_once(shape, size):
    game = Kalah(*shape)
    assert game_graph_size(game, game.initial_state()) == size


def test_tic_tac_toe_has_5478_reachable_positions():
    assert game_graph_size(TicTacToe(), TicTacToe().initial_state()) == 5478


def test_the_canonical_strategy_plays_the_first_value_keeping_move():
    game = TicTacToe()
    s = _ttt((0, 1), (2, 3))
    choice = canonical_strategy(game, s, 0)
    assert choice[game.state_key(s)] == 4


def _certified(game, root, player, choice):
    r = certify(game, root, player, lambda states: [choice[game.state_key(s)] for s in states],
                game.position_value)
    return r["certified"] and r["complete_game"]


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 3), (4, 1)])
def test_both_strategies_certify_over_whole_kalah_games(shape):
    game = Kalah(*shape)
    root = game.initial_state()
    for choice in (canonical_strategy(game, root, 0), min_tree_strategy(game, root, 0)["choice"]):
        assert _certified(game, root, 0, choice)


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 6), (3, 3), (4, 1)])
def test_minimising_the_tree_never_needs_more_decisions_than_the_canonical_tree(shape):
    game = Kalah(*shape)
    root = game.initial_state()
    canonical = canonical_strategy(game, root, 0)
    best = min_tree_strategy(game, root, 0)
    assert best["tree"] <= _tree_decisions(game, root, 0, canonical)
    assert decisions(game, root, 0, best["choice"]) <= best["tree"]


def _tree_decisions(game, root, player, choice):
    if game.is_terminal(root):
        return 0
    if game.current_player(root) == player:
        return 1 + _tree_decisions(game, game.step(root, choice[game.state_key(root)]), player, choice)
    return sum(_tree_decisions(game, game.step(root, b), player, choice) for b in game.legal_actions(root))


def test_the_minimal_tree_is_the_smallest_over_every_value_keeping_choice():
    game = TicTacToe()
    s = _ttt((0, 1), (2, 3))
    assert min_tree_strategy(game, s, 0)["tree"] == 5
    s2 = _ttt((0,), (1,))
    best = min_tree_strategy(game, s2, 0)
    assert best["tree"] == _tree_decisions(game, s2, 0, best["choice"])
    assert _certified(game, s2, 0, best["choice"])


def test_decisions_count_each_position_of_the_strategy_once():
    game = TicTacToe()
    s = _ttt((0, 1), (2, 3))
    assert decisions(game, s, 0, min_tree_strategy(game, s, 0)["choice"]) == 5
    assert decisions(game, s, 0, canonical_strategy(game, s, 0)) == 7
    assert decisions(game, game.initial_state(), 0, canonical_strategy(game, game.initial_state(), 0)) < \
        _tree_decisions(game, game.initial_state(), 0, canonical_strategy(game, game.initial_state(), 0))


def test_a_strategy_for_the_second_player_is_built_the_same_way():
    game = Kalah(2, 2)
    root = game.initial_state()
    choice = min_tree_strategy(game, root, 1)["choice"]
    assert _certified(game, root, 1, choice)
