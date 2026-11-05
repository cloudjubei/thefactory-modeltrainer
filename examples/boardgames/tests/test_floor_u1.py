"""Direct tests for harness/floor_u1.py — U1, choosing among G4's rebuilds by local search: one rebuild or every
rebuild playing one map in or out at a time, from none kept and from all kept, never worse than G4's whole game."""
from __future__ import annotations

import pytest

import harness.floor_g4 as floor_g4
import harness.floor_j1 as floor_j1
import harness.floor_u1 as floor_u1
from harness.floor_u1 import u1_choose, u1_size
from tests.test_floor_g4 import ASSEMBLY, CONTROL, G3, JUDGED, _arm, _earlier, _key, _size
from tests.test_floor_g4 import _cost as _g4_cost


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


def _cost(monkeypatch):
    _g4_cost(monkeypatch)
    monkeypatch.setattr(floor_u1, "map_bits", floor_g4.map_bits)


def _toy(saves: dict, plays: dict, cost: dict):
    def size(held):
        return 100 - sum(saves[k] for k in held) + sum(cost[m] for m in set().union(*(plays[k] for k in held)))

    return size


def test_a_single_rebuild_goes_in_or_out_while_it_shrinks_the_game():
    plays = {"a": frozenset({1}), "b": frozenset({2})}
    size = _toy({"a": 20, "b": 3}, plays, {1: 5, 2: 9})
    assert u1_choose(size, plays, frozenset()) == (frozenset({"a"}), 85)
    assert u1_choose(size, plays, frozenset({"a", "b"})) == (frozenset({"a"}), 85)


def test_the_rebuilds_sharing_a_map_go_in_together():
    plays = {"a": frozenset({1}), "b": frozenset({1}), "c": frozenset({1})}
    size = _toy({"a": 8, "b": 8, "c": 7}, plays, {1: 13})
    assert u1_choose(size, plays, frozenset()) == (frozenset({"a", "b", "c"}), 90)


def test_the_rebuilds_sharing_a_map_go_out_together():
    plays = {"a": frozenset({1}), "b": frozenset({1, 2}), "c": frozenset({2})}
    size = _toy({"a": 3, "b": 3, "c": 30}, plays, {1: 13, 2: 5})
    assert u1_choose(size, plays, frozenset({"a", "b", "c"})) == (frozenset({"c"}), 75)


def test_a_map_played_by_one_rebuild_is_no_group_move():
    calls = []
    plays = {"a": frozenset({1}), "b": frozenset({2})}
    toy = _toy({"a": 1, "b": 1}, plays, {1: 5, 2: 5})

    def size(held):
        calls.append(held)
        return toy(held)

    u1_choose(size, plays, frozenset())
    assert sorted(map(sorted, calls)) == [[], ["a"], ["b"]]


def test_the_best_move_is_taken_first_and_ties_change_nothing():
    table = {frozenset(): 100, frozenset("a"): 99, frozenset("b"): 90, frozenset("ab"): 101}
    plays = {"a": frozenset({1}), "b": frozenset({2})}
    assert u1_choose(table.__getitem__, plays, frozenset()) == (frozenset({"b"}), 90)
    plays = {"a": frozenset({1})}
    assert u1_choose(_toy({"a": 7}, plays, {1: 7}), plays, frozenset()) == (frozenset(), 100)


def test_the_search_stops_where_no_move_shrinks_the_game():
    plays = {"a": frozenset({1, 2}), "b": frozenset({1}), "c": frozenset({2})}
    size = _toy({"a": 8, "b": 8, "c": 8}, plays, {1: 13, 2: 13})
    assert u1_choose(size, plays, frozenset()) == (frozenset(), 100)
    assert u1_choose(size, plays, frozenset({"a", "b", "c"})) == (frozenset({"a", "b", "c"}), 102)


def test_the_whole_game_from_the_better_start_with_the_ply_6_builds_chosen_again(monkeypatch):
    _cost(monkeypatch)
    r = u1_size(_arm(), *_args())
    bits = 4 + 8 + 90 + 200 + 50 + 1 + 40 + 60 + 13 + 200
    assert r == {"verdict": "measured", "bits": bits, "g4_bits": bits, "kept": 2, "from_none": bits,
                 "from_all": bits}


def test_a_rebuild_pair_g4_keeps_because_neither_goes_back_alone_goes_back_together(monkeypatch):
    _cost(monkeypatch)
    arm = _arm(bits=(94, 294, 45), leaves=((1, 5), (5,), (2,)))
    g4 = _size(arm=arm)
    r = u1_size(arm, *_args())
    assert (g4["bits"], g4["rebuilt"]) == (CONTROL - 4, 3)
    assert (r["bits"], r["g4_bits"], r["kept"], r["from_none"], r["from_all"]) == (CONTROL - 5, CONTROL - 4, 1,
                                                                                   CONTROL - 5, CONTROL - 5)


def test_a_rebuild_pair_comes_in_together(monkeypatch):
    _cost(monkeypatch)
    arm = _arm(bits=(92, 292, 50), leaves=((1, 5), (5,), (2,)))
    r = u1_size(arm, *_args())
    assert (r["bits"], r["kept"], r["from_none"]) == (CONTROL - 3, 2, CONTROL - 3)


def test_the_smaller_of_the_two_starts_is_kept(monkeypatch):
    _cost(monkeypatch)
    real = floor_u1.u1_choose

    def worse_from(start_empty):
        def choose(size, plays, start):
            held, bits = real(size, plays, start)
            return (frozenset(), bits + 50) if (not start) == start_empty else (held, bits)

        return choose

    for start_empty in (True, False):
        monkeypatch.setattr(floor_u1, "u1_choose", worse_from(start_empty))
        r = u1_size(_arm(bits=(94, 294, 45), leaves=((1, 5), (5,), (2,))), *_args())
        assert (r["bits"], r["kept"]) == (CONTROL - 5, 1)
        assert (r["from_none"], r["from_all"])[not start_empty] == CONTROL - 5 + 50


def test_only_rebuilds_below_positions_keeping_g3_s_pieces_count(monkeypatch):
    _cost(monkeypatch)
    real = floor_u1.v2_assemble

    def last_takes_0(rows, a):
        chosen, bits = real(rows, a)
        return ({**chosen, _key(0): 0}, bits) if a is not ASSEMBLY else (chosen, bits)

    monkeypatch.setattr(floor_u1, "v2_assemble", last_takes_0)
    assert u1_size(_arm(), *_args())["kept"] == 1

def test_the_ply_6_builds_chosen_again_only_where_that_shrinks_the_game(monkeypatch):
    _cost(monkeypatch)
    real = floor_u1.v2_assemble
    monkeypatch.setattr(floor_u1, "v2_assemble",
                        lambda rows, a: real(rows, a) if a is ASSEMBLY else (lambda c, b: (c, b + 7))(*real(rows, a)))
    r = u1_size(_arm(bits=(94, 294, 45), leaves=((1, 5), (5,), (2,))), *_args())
    assert (r["bits"], r["kept"]) == (CONTROL - 5, 1)


def test_the_game_is_never_worse_than_g4_s(monkeypatch):
    _cost(monkeypatch)
    monkeypatch.setattr(floor_u1, "u1_choose", lambda size, plays, start: (frozenset(), size(frozenset())))
    r = u1_size(_arm(), *_args())
    assert r["bits"] == r["g4_bits"] == 4 + 8 + 90 + 200 + 50 + 1 + 40 + 60 + 13 + 200 and r["kept"] == 3


def test_a_run_g4_does_not_judge_is_not_run(monkeypatch):
    _cost(monkeypatch)
    assert u1_size({**_arm(), "measurement_fingerprint": "other"}, *_args()) == {"verdict": "not_run"}


def test_an_unfinished_rebuild_stays_g3_s(monkeypatch):
    _cost(monkeypatch)
    arm = _arm(bits=(92, 292, 50), leaves=((1, 5), (5,), (2,)), incomplete=(1,))
    assert u1_size(arm, *_args())["bits"] == CONTROL


def _args():
    return _earlier(), G3, ASSEMBLY, JUDGED
