"""Direct tests for harness/reference_coding.py — exact codes for which map each leaf of a strategy plays, read in the
order a decoder replays the strategy: the builder's fixed width over the whole map list, or the library maps used
named once (how many, gamma-coded, and which) and every leaf a move-to-front rank, the first use of a build's own map
a single NEW symbol."""
from __future__ import annotations

import math

import pytest

from harness.reference_coding import CODES, code_bits, gamma_bits, walk_leaves


def _named(shared, used):
    return gamma_bits(used + 1) + math.ceil(math.log2(math.comb(shared + 1, used)))


def test_gamma_codes_a_positive_count_in_twice_its_length_less_one():
    assert [gamma_bits(n) for n in (1, 2, 3, 4, 7, 8, 1000)] == [1, 3, 3, 5, 5, 7, 19]
    with pytest.raises(ValueError):
        gamma_bits(0)


def test_the_fixed_code_is_the_builder_s_width_over_the_whole_list_per_leaf():
    assert code_bits("fixed", [0, 3, 3, 1], 9, 2) == 4 * 4
    assert code_bits("fixed", [0, 0], 1, 0) == 0
    assert code_bits("fixed", [1, 2], 3, 0) == 2 * 2


def test_a_build_s_own_maps_cost_one_new_symbol_at_first_use_and_their_rank_after():
    refs = [3, 4, 3, 3]
    table = _named(2, 0)
    assert code_bits("first_use", refs, 5, 2) == table + 1 + 1 + 3 + 1


def test_library_maps_are_named_once_and_ranked_among_the_used():
    refs = [1, 2, 1, 1]
    table = _named(2, 2)
    assert code_bits("first_use", refs, 3, 2) == table + 3 + 3 + 3 + 1


def test_the_empty_map_counts_among_the_named_ones():
    assert code_bits("first_use", [0], 1, 0) == _named(0, 1) + 3
    assert code_bits("first_use", [0, 1], 2, 0) == _named(0, 1) + 3 + 3


def test_a_seen_map_moves_to_the_front():
    refs = [3, 4, 5, 3, 3]
    assert code_bits("first_use", refs, 6, 2) == _named(2, 0) + 1 + 1 + 1 + 5 + 1


def test_an_own_map_no_leaf_plays_cannot_be_coded_by_first_use():
    with pytest.raises(ValueError):
        code_bits("first_use", [3], 5, 2)
    with pytest.raises(ValueError):
        code_bits("first_use", [3, 4, 5], 5, 2)
    with pytest.raises(ValueError):
        code_bits("first_use", [3, 4, -1], 5, 2)
    with pytest.raises(ValueError):
        code_bits("nope", [0], 1, 0)


def test_the_codes_and_how_many_library_maps_are_named():
    assert CODES == ("fixed", "first_use")
    assert (_named(2, 0), _named(2, 2), _named(0, 1), _named(1999, 1)) == (1, 5, 3, 3 + 11)


class Tree:
    """A two-player game on integers: playing a at s reaches 10 * s + a, the moves, the player to move (0 unless
    listed) and the states that end the game given by tables."""

    def __init__(self, moves, player, terminal):
        self.moves, self.player, self.terminal = moves, player, terminal

    def state_key(self, s):
        return s

    def step(self, s, a):
        return 10 * s + a

    def is_terminal(self, s):
        return s in self.terminal

    def current_player(self, s):
        return self.player.get(s, 0)

    def legal_actions(self, s):
        return self.moves.get(s, [])


def test_leaves_are_read_depth_first_replies_in_order_each_position_once():
    game = Tree(moves={11: [1, 2, 3], 21: [1]}, player={11: 1, 21: 1}, terminal={113})
    nodes = {1: {"move": 1}, 111: {"leaf": 4}, 112: {"move": 1}, 1121: {"leaf": 2}, 2: {"leaf": 9}}
    assert walk_leaves(game, 1, nodes) == [4, 2]


def test_a_move_keeping_the_turn_leads_straight_on_and_a_transposition_is_read_once():
    class Joined(Tree):
        def step(self, s, a):
            return 7 if (s, a) in {(51, 1), (51, 2)} else 10 * s + a

    game = Joined(moves={51: [1, 2]}, player={51: 1}, terminal=set())
    nodes = {4: {"move": 2}, 42: {"move": 3}, 423: {"move": 1}, 4231: {"leaf": 6}}
    assert walk_leaves(game, 4, nodes) == [6]
    nodes = {5: {"move": 1}, 7: {"leaf": 3}}
    assert walk_leaves(game, 5, nodes) == [3]


def test_a_terminal_move_ends_the_line_and_a_missing_node_is_refused():
    game = Tree(moves={}, player={}, terminal={11})
    assert walk_leaves(game, 1, {1: {"move": 1}}) == []
    game = Tree(moves={11: [1]}, player={11: 1}, terminal=set())
    with pytest.raises(KeyError):
        walk_leaves(game, 1, {1: {"move": 1}})

