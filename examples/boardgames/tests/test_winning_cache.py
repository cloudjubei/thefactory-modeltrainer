"""Direct tests for harness/winning_cache.py — the oracle's winning moves remembered across a build's searches, in a
cache emptied whenever it would outgrow its limit."""
from __future__ import annotations

from harness.winning_cache import CachedWinning


class _Oracle:
    def __init__(self):
        self.asked = []

    def __call__(self, states):
        self.asked.append(list(states))
        return [{s % 7} for s in states]


def test_answers_match_the_oracle_and_repeats_are_not_asked_again():
    oracle = _Oracle()
    cached = CachedWinning(oracle, key=lambda s: s, limit=100)
    assert cached([3, 10]) == [{3}, {3}]
    assert cached([10, 4, 3]) == [{3}, {4}, {3}]
    assert oracle.asked == [[3, 10], [4]]


def test_a_batch_asks_each_new_position_once():
    oracle = _Oracle()
    assert CachedWinning(oracle, key=lambda s: s, limit=100)([5, 5, 6]) == [{5}, {5}, {6}]
    assert oracle.asked == [[5, 6]]


def test_positions_are_told_apart_by_their_key():
    oracle = _Oracle()
    cached = CachedWinning(oracle, key=lambda s: s % 10, limit=100)
    cached([1])
    assert cached([11]) == [{1}] and oracle.asked == [[1]]


def test_the_cache_is_emptied_before_it_outgrows_its_limit_and_answers_stay_right():
    oracle = _Oracle()
    cached = CachedWinning(oracle, key=lambda s: s, limit=3)
    cached([1, 2, 3])
    assert cached([1, 4]) == [{1}, {4}]
    assert len(cached.cache) <= 3 and cached.cleared == 1
    cached([2])
    assert oracle.asked[-1] == [2]


def test_nothing_is_asked_for_an_empty_batch():
    oracle = _Oracle()
    assert CachedWinning(oracle, key=lambda s: s, limit=3)([]) == [] and oracle.asked == []
