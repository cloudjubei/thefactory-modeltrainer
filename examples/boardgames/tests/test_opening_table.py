"""Direct tests for harness/opening_table.py — the hybrid's EXCEPTION TABLE: walk the strategy's own tree for one
player, and at that player's positions before a horizon store an exactly optimal move wherever the strategy's move is
not optimal; the hybrid plays the table where it has an entry and the strategy everywhere else."""
from __future__ import annotations

import random

import pytest

from games.tictactoe import TicTacToe
from harness.certify import certify
from harness.coverage import move_values
from harness.opening_table import exception_table, full_table, hybrid_chooser

G = TicTacToe()
ROOT = G.initial_state(random.Random(0))


def _values(states):
    return [move_values(G, s) for s in states]


def _first_legal(states):
    return [G.legal_actions(s)[0] for s in states]


def _perfect(states):
    out = []
    for s in states:
        v = move_values(G, s)
        out.append(max(v, key=lambda a: (v[a], -a)))
    return out


def test_a_perfect_strategy_needs_no_table():
    table, stats = exception_table(G, ROOT, 0, _perfect, _values, horizon=9)
    assert table == {} and stats["overridden"] == 0 and stats["positions"] > 0


def test_every_entry_is_an_optimal_move_where_the_strategy_s_move_was_not():
    table, stats = exception_table(G, ROOT, 0, _first_legal, _values, horizon=9)
    assert table and stats["overridden"] == len(table)
    for key, move in table.items():
        s = next(st for st in _walk_all() if G.state_key(st) == key)
        v = move_values(G, s)
        assert v[move] == max(v.values()) and v[G.legal_actions(s)[0]] < max(v.values())


def _walk_all():
    seen, level, out = set(), [ROOT], []
    while level:
        nxt = []
        for s in level:
            k = G.state_key(s)
            if k in seen or G.is_terminal(s):
                continue
            seen.add(k)
            out.append(s)
            nxt += [G.step(s, a) for a in G.legal_actions(s)]
        level = nxt
    return out


def test_the_hybrid_with_a_whole_game_table_certifies_and_the_strategy_alone_does_not():
    table, _stats = exception_table(G, ROOT, 0, _first_legal, _values, horizon=9)
    value = lambda s: max(move_values(G, s).values())
    assert not certify(G, ROOT, 0, _first_legal, value)["certified"]
    assert certify(G, ROOT, 0, hybrid_chooser(G, table, _first_legal), value)["certified"]


@pytest.mark.parametrize("horizon,plies", [(1, {0}), (2, {0}), (3, {0, 2}), (4, {0, 2}), (5, {0, 2, 4})])
def test_the_table_covers_only_the_player_s_positions_before_the_horizon(horizon, plies):
    table, stats = exception_table(G, ROOT, 0, _first_legal, _values, horizon=horizon)
    assert {sum(1 for c in board if c) for board, _side in table} <= plies
    assert all(side == 0 for _board, side in table)
    assert set(stats["positions_by_ply"]) <= {str(p) for p in plies}


def test_the_walk_follows_the_hybrid_s_own_moves_so_the_tree_is_the_one_it_will_play():
    table, stats = exception_table(G, ROOT, 0, _first_legal, _values, horizon=3)
    first = G.step(ROOT, table.get(G.state_key(ROOT), 0))
    second_ply = {G.state_key(G.step(first, b)) for b in G.legal_actions(first)}
    assert stats["positions_by_ply"]["2"] == len(second_ply)


def test_the_chooser_falls_back_to_the_strategy_off_the_table():
    choose = hybrid_chooser(G, {G.state_key(ROOT): 4}, _first_legal)
    other = G.step(ROOT, 0)
    assert choose([ROOT, other]) == [4, G.legal_actions(other)[0]]


def test_a_horizon_below_one_is_refused():
    with pytest.raises(ValueError, match="horizon"):
        exception_table(G, ROOT, 0, _first_legal, _values, horizon=0)


class _Cycle:
    """Four positions, players alternating by parity, every move leading back round the cycle — positions recur."""

    def state_key(self, s):
        return s

    def is_terminal(self, s):
        return False

    def current_player(self, s):
        return s % 2

    def legal_actions(self, s):
        return [0, 1]

    def step(self, s, a, rng):
        return (s + 1 + a) % 4


def test_a_position_reached_again_at_a_later_ply_is_counted_and_tabled_once():
    table, stats = exception_table(_Cycle(), 0, 0, lambda states: [0] * len(states),
                                   lambda states: [{0: 0, 1: 1} for _ in states], horizon=12)
    assert stats["positions"] == 2 and stats["overridden"] == 2 and table == {0: 1, 2: 1}


def test_a_full_table_stores_the_first_optimal_move_at_every_player_position_before_the_horizon():
    table, frontier = full_table(G, ROOT, 0, _values, horizon=3)
    assert len(table) == 1 + 8
    for (board, side), move in table.items():
        assert side == 0 and sum(1 for c in board if c) in (0, 2)
        s = next(st for st in _walk_all() if G.state_key(st) == (board, side))
        v = move_values(G, s)
        assert move == next(a for a in G.legal_actions(s) if v[a] == max(v.values()))


def test_the_frontier_is_every_unfinished_position_at_the_horizon_on_the_table_s_tree():
    table, frontier = full_table(G, ROOT, 0, _values, horizon=3)
    expected = set()
    for s2 in (G.step(G.step(ROOT, table[G.state_key(ROOT)]), b) for b in G.legal_actions(
            G.step(ROOT, table[G.state_key(ROOT)]))):
        child = G.step(s2, table[G.state_key(s2)])
        if not G.is_terminal(child):
            expected.add(G.state_key(child))
    assert {G.state_key(s) for s in frontier} == expected and len(frontier) == len(expected)
    assert all(sum(1 for c in s.board if c) == 3 and not G.is_terminal(s) for s in frontier)


def test_a_full_table_horizon_below_one_is_refused():
    with pytest.raises(ValueError, match="horizon"):
        full_table(G, ROOT, 0, _values, horizon=0)


def test_games_the_table_wins_before_the_horizon_are_not_part_of_the_frontier():
    table, frontier = full_table(G, ROOT, 0, _values, horizon=5)
    reached, level = [], [ROOT]
    for _ply in range(5):
        nxt = []
        for s in level:
            if G.is_terminal(s):
                continue
            moves = [table[G.state_key(s)]] if G.current_player(s) == 0 else G.legal_actions(s)
            nxt += [G.step(s, a) for a in moves]
        level = nxt
    finished = {G.state_key(s) for s in level if G.is_terminal(s)}
    assert finished and not finished & {G.state_key(s) for s in frontier}
    assert {G.state_key(s) for s in frontier} == {G.state_key(s) for s in level if not G.is_terminal(s)}
