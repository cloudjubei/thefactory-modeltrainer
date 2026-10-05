"""Direct tests for harness/description_length.py — the north star's "smallest total description" of a strategy: the
net's parameters at a declared precision plus a table whose entries each name one position of the strategy's own tree
and one move."""
from __future__ import annotations

import pytest

from harness.description_length import entry_bits, hybrid_bits, ranking


@pytest.mark.parametrize("positions,actions,bits", [(1, 7, 3), (2, 7, 4), (1024, 7, 13), (1025, 7, 14),
                                                    (300, 9, 13), (5, 2, 4), (5, 1, 3)])
def test_an_entry_names_one_of_the_tree_s_positions_and_one_move(positions, actions, bits):
    assert entry_bits(positions, actions) == bits


def test_entry_bits_refuse_an_empty_tree_or_a_game_without_moves():
    with pytest.raises(ValueError, match="positions"):
        entry_bits(0, 7)
    with pytest.raises(ValueError, match="actions"):
        entry_bits(10, 0)


def test_a_hybrid_is_its_net_at_the_declared_precision_plus_its_table():
    d = hybrid_bits(params=1000, bits_per_param=16, entries=10, positions=1024, actions=7)
    assert d == {"net": 16000, "table": 130, "total": 16130}


def test_a_table_alone_and_a_net_alone_are_the_two_extremes_of_the_same_sum():
    assert hybrid_bits(params=0, bits_per_param=32, entries=300, positions=300, actions=7)["total"] == 300 * 12
    assert hybrid_bits(params=500, bits_per_param=8, entries=0, positions=300, actions=7) == {"net": 4000, "table": 0,
                                                                                            "total": 4000}


def test_hybrid_bits_refuse_a_table_larger_than_the_tree():
    with pytest.raises(ValueError, match="entries"):
        hybrid_bits(params=10, bits_per_param=8, entries=301, positions=300, actions=7)


def test_variants_are_ranked_smallest_total_first_and_only_certified_ones_count():
    variants = {"table only": {"total": 4000, "certified": True}, "hybrid": {"total": 2500, "certified": True},
                "net only": {"total": 900, "certified": False}}
    assert ranking(variants) == [("hybrid", 2500), ("table only", 4000)]
