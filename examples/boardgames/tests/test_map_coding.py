"""Direct tests for harness/map_coding.py — exact codes for a priority map, given the cells empty at its position (the
decoder knows them): which cells carry a level, and their levels, under the rule that reads only the levels' ORDER
(harness.steady_state.choose passes over empty levels, so renumbering the used levels 0..u-1 changes no move)."""
from __future__ import annotations

import math

import pytest

from harness.map_coding import CODES, best_bits, code_bits, ranked

FIVE = {3: 0, 7: 0, 9: 2, 11: 5, 12: 5}


def test_the_bitmap_code_is_the_builder_s_one_bit_an_empty_cell_and_three_a_level():
    assert code_bits("bitmap", FIVE, 30, 8) == 30 + 3 * 5
    assert code_bits("bitmap", {}, 30, 8) == 30


def test_choosing_the_levelled_cells_costs_their_count_and_which_of_the_empty_cells():
    assert code_bits("chosen", FIVE, 30, 8) == math.ceil(math.log2(math.comb(30, 5))) + math.ceil(math.log2(31)) + 15


def test_ranked_levels_keep_their_order_and_drop_the_gaps():
    assert ranked(FIVE) == {3: 0, 7: 0, 9: 1, 11: 2, 12: 2}
    assert ranked({}) == {}


def test_ranked_levels_cost_the_count_of_levels_used_and_a_rank_a_cell():
    head = math.ceil(math.log2(math.comb(30, 5))) + math.ceil(math.log2(31))
    assert code_bits("ranks", FIVE, 30, 8) == head + 3 + 5 * 2
    assert code_bits("ranks", {1: 4, 2: 4}, 30, 8) == math.ceil(math.log2(math.comb(30, 2))) + 5 + 3 + 0


def test_level_counts_cost_how_the_cells_split_over_the_used_levels_and_which_cell_has_which():
    head = math.ceil(math.log2(math.comb(30, 5))) + math.ceil(math.log2(31)) + 3
    split = math.ceil(math.log2(math.comb(5 - 1, 3 - 1)))
    which = math.ceil(math.log2(math.factorial(5) // (math.factorial(2) * math.factorial(1) * math.factorial(2))))
    assert code_bits("counts", FIVE, 30, 8) == head + split + which


def test_an_empty_map_costs_only_the_cell_count_in_the_chosen_codes():
    assert code_bits("ranks", {}, 30, 8) == code_bits("counts", {}, 30, 8) == code_bits("chosen", {}, 30, 8) == 5


def test_the_best_code_is_the_cheapest_plus_a_selector():
    costs = {c: code_bits(c, FIVE, 30, 8) for c in CODES}
    assert best_bits(FIVE, 30, 8) == (min(costs, key=costs.get), min(costs.values()) + 2)
    assert CODES == ("bitmap", "chosen", "ranks", "counts")


def test_an_unknown_code_or_more_levelled_cells_than_empty_ones_is_refused():
    with pytest.raises(ValueError, match="code"):
        code_bits("huffman", FIVE, 30, 8)
    with pytest.raises(ValueError, match="empty"):
        code_bits("chosen", FIVE, 4, 8)
