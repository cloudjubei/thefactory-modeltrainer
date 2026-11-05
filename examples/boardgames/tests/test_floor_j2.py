"""Direct tests for harness/floor_j2.py — J2, the whole game assembled by transposition group: ply-6 positions sharing
a ply-8 position below them choose together, every subset of a group's positions with builds tried at once."""
from __future__ import annotations

import pytest

from harness.floor_j1 import j1_bits, j1_choose
from harness.floor_j2 import j2_choose, j2_groups

NONE = {"moves": 0, "trivial": 0}


def test_positions_sharing_a_ply_8_position_form_a_group_transitively():
    below = {"s": ["d"], "p": ["a", "b"], "q": ["b", "c"], "r": ["c"], "t": []}
    assert j2_groups(below) == [["p", "q", "r"], ["s"], ["t"]]


SHARED = {"p": ["b"], "q": ["b"], "r": ["b"]}
SHARED_BASE = {"b": (3000, frozenset())}
EACH = {k: [(k, (990, frozenset()))] for k in SHARED}


def test_a_build_sharing_its_ply_8_positions_pays_only_with_its_whole_group():
    assert j1_choose(NONE, SHARED, SHARED_BASE, EACH, {}) == ({}, 3012)
    chosen, bits = j2_choose(NONE, SHARED, SHARED_BASE, EACH, {}, {})
    assert chosen == {"p": "p", "q": "q", "r": "r"} and bits == 2970


def test_part_of_a_group_swaps_when_that_alone_shrinks_it():
    below = {"p": ["a", "b"], "q": ["b"]}
    base = {"a": (500, frozenset()), "b": (100, frozenset())}
    candidates = {"p": [("p", (50, frozenset()))], "q": [("q", (10_000, frozenset()))]}
    assert j2_choose(NONE, below, base, candidates, {}, {}) == ({"p": "p"}, 4 + 100 + 50)


def test_the_start_stands_where_no_group_improves_on_it():
    below, base = {"p": ["a"]}, {"a": (1000, frozenset())}
    candidates = {"p": [("x", (500, frozenset())), ("y", (500, frozenset()))]}
    assert j2_choose(NONE, below, base, candidates, {}, {}) == ({"p": "x"}, 500)
    bits = j1_bits(NONE, below, base, {"p": (500, frozenset())}, {})["bits"]
    assert j2_choose(NONE, below, base, candidates, {}, {"p": "y"}) == ({"p": "y"}, bits)


def test_a_group_s_change_keeps_the_other_groups_choices():
    below = {**SHARED, "s": ["d"]}
    base = {**SHARED_BASE, "d": (100, frozenset())}
    candidates = {**EACH, "s": [("s", (20, frozenset()))]}
    chosen, bits = j2_choose(NONE, below, base, candidates, {}, {"s": "s"})
    assert chosen == {"p": "p", "q": "q", "r": "r", "s": "s"} and bits == 2970 + 20


def test_ties_keep_the_current_choice():
    candidates = {k: [(k, (1004, frozenset()))] for k in SHARED}
    assert j2_choose(NONE, SHARED, SHARED_BASE, candidates, {}, {}) == ({}, 3012)


def _maps_case(other_plays, own_plays=(), other_built=False):
    below = {"p": ["a"], "o": ["m"]}
    base = {"a": (1000, frozenset(own_plays)), "m": (10, frozenset(other_plays))}
    candidates = {"p": [("cheap", (100, frozenset({1}))), ("plain", (300, frozenset()))]}
    if other_built:
        candidates["o"] = [("o", (5, frozenset()))]
    return j2_choose(NONE, below, base, candidates, {1: 500}, {"o": "o"} if other_built else {})


def test_a_swapped_position_takes_the_build_whose_maps_the_rest_already_pays():
    assert _maps_case({1})[0] == {"p": "cheap"}
    assert _maps_case(set())[0] == {"p": "plain"}


def test_maps_only_the_swapping_positions_or_built_ones_would_play_are_not_paid():
    assert _maps_case(set(), own_plays={1})[0] == {"p": "plain"}
    assert _maps_case({1}, other_built=True) == ({"o": "o", "p": "plain"}, 305)


def test_groups_are_tried_again_while_any_improves():
    below = {"p": ["a"], "q": ["c"]}
    base = {"a": (600, frozenset()), "c": (1000, frozenset())}
    candidates = {"p": [("p", (200, frozenset({1})))], "q": [("q", (100, frozenset({1})))]}
    chosen, bits = j2_choose(NONE, below, base, candidates, {1: 500}, {})
    assert chosen == {"p": "p", "q": "q"} and bits == 200 + 100 + 500


def test_a_group_with_too_many_positions_with_builds_is_refused():
    with pytest.raises(ValueError, match="3 positions with builds"):
        j2_choose(NONE, SHARED, SHARED_BASE, EACH, {}, {}, max_swapped=2)
    assert j2_choose(NONE, SHARED, SHARED_BASE, EACH, {}, {}, max_swapped=3)[1] == 2970


def test_maps_another_group_s_build_plays_are_paid():
    below = {"p": ["a"], "o": ["m"]}
    base = {"a": (1000, frozenset()), "m": (2000, frozenset())}
    candidates = {"p": [("cheap", (100, frozenset({1}))), ("plain", (300, frozenset()))],
                  "o": [("o", (5, frozenset({1})))]}
    assert j2_choose(NONE, below, base, candidates, {1: 500}, {"o": "o"}) == ({"o": "o", "p": "cheap"}, 605)
