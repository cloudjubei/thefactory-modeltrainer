"""Direct tests for harness/floor_u2.py — U2, the whole game chosen again from every certified piece the runs hold:
each position takes one of its pieces, and a move changes one position or every position sharing a map."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_u2 as floor_u2
import harness.floor_x2 as floor_x2
import harness.floor_x3 as floor_x3
from harness.floor_u2 import SPEC, u2_choose, u2_game, u2_size
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC as X4
from tests.test_floor_g4 import ASSEMBLY, BUILT, G3, LIB, _key, _kid, _row, _run
from tests.test_floor_x2 import CONTROL as Z_CONTROL, JUDGED as X2_JUDGED, LIBRARY, N, READS, _arm as _x2_arm
from tests.test_floor_x3 import CONTROL as X2_GAME, JUDGED as X3_JUDGED, _arm as _x3_arm, _x2, _z
from tests.test_floor_x4 import JUDGED as X4_JUDGED
from tests.test_floor_z1 import FOUND, G4_LIBRARY, _earlier, _g4

GAME = X2_GAME - 15
JUDGED = {**SPEC, "x3_held": [11, 12], "control_bits": GAME}


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


@pytest.fixture(autouse=True)
def _costs(monkeypatch):
    costs = (40, 60, 11, 12, 13, 14, 15)

    def cost(levels, empty, level_bits):
        return costs[[{int(c): k for c, k in m.items()} for m, _ in LIBRARY].index(levels)]

    for module in (floor_x2, floor_x3, floor_u2):
        monkeypatch.setattr(module, "map_bits", cost)


def _toy(cost: dict, options: dict):
    def size(state):
        maps = set().union(*(options[k][label][1] for k, label in state.items() if options[k][label]))
        return sum(options[k][label][0] for k, label in state.items() if options[k][label]) + sum(cost[m] for m in maps)

    return size


def _x4(bits=60, leaves=(N + 2,), complete=True, library=READS):
    return _run([_row(12, bits, leaves=leaves, library=library, complete=complete)], X4_JUDGED), X4_JUDGED


def _x3(arm=None):
    return arm or _x3_arm(), X3_JUDGED


def _game(x4=None, x3=None, spec=JUDGED, g4=None, x2=None, earlier=None):
    return u2_game(x4 or _x4(), x3 or _x3(), x2 or _x2(), _z(), earlier or _earlier(), g4 or _g4(), G3, ASSEMBLY, spec)


def _size(x4=None, x3=None, spec=JUDGED, g4=None):
    return u2_size(x4 or _x4(), x3 or _x3(), _x2(), _z(), _earlier(), g4 or _g4(), G3, ASSEMBLY, spec)


def test_the_spec_names_x3_s_game_and_the_bars():
    assert SPEC["control_bits"] == 166_623 and len(SPEC["x3_held"]) == 71 and SPEC["x3_held"] == sorted(SPEC["x3_held"])
    assert set(SPEC["x3_held"]) <= set(X3["indices"]) and (SPEC["support"], SPEC["refute"]) == ((199, 200), (999, 1000))
    assert X4["control_bits"] == SPEC["control_bits"]


def test_one_position_moves_to_its_smaller_piece():
    options = {"a": {"x": (9, frozenset()), "y": (5, frozenset())}, "b": {"x": (3, frozenset())}}
    assert u2_choose(_toy({}, options), options, {"a": "x", "b": "x"}) == ({"a": "y", "b": "x"}, 8)


def test_a_smaller_piece_that_pays_a_map_alone_stays_out():
    options = {"a": {"x": (9, frozenset()), "y": (5, frozenset({1}))}}
    assert u2_choose(_toy({1: 7}, options), options, {"a": "x"}) == ({"a": "x"}, 9)


def test_the_positions_sharing_a_new_map_go_over_together():
    options = {k: {"x": (9, frozenset()), "y": (5, frozenset({1}))} for k in "abc"}
    assert u2_choose(_toy({1: 7}, options), options, dict.fromkeys("abc", "x")) == (dict.fromkeys("abc", "y"), 22)


def test_the_positions_paying_a_map_leave_it_together():
    options = {k: {"x": (6, frozenset({1})), "y": (9, frozenset())} for k in "ab"}
    assert u2_choose(_toy({1: 7}, options), options, dict.fromkeys("ab", "x")) == (dict.fromkeys("ab", "y"), 18)


def test_a_group_move_takes_each_position_s_smallest_piece_with_the_map():
    options = {k: {"x": (9, frozenset()), "big": (8, frozenset({1})), "y": (5, frozenset({1}))} for k in "ab"}
    size = _toy({1: 7}, options)
    assert u2_choose(size, options, dict.fromkeys("ab", "x")) == (dict.fromkeys("ab", "y"), 17)


def test_a_group_move_out_takes_each_position_s_smallest_piece_without_the_map():
    options = {k: {"x": (6, frozenset({1})), "big": (11, frozenset()), "y": (9, frozenset())} for k in "ab"}
    assert u2_choose(_toy({1: 7}, options), options, dict.fromkeys("ab", "x")) == (dict.fromkeys("ab", "y"), 18)


def test_a_group_move_never_drops_a_position_s_piece_for_none():
    options = {k: {"x": (6, frozenset({1})), None: None} for k in "ab"}
    calls = []

    def size(state):
        calls.append(dict(state))
        return 100 if None in state.values() else 50

    u2_choose(size, options, dict.fromkeys("ab", "x"))
    assert {"a": None, "b": None} not in calls and len(calls) == 3


def test_a_map_played_at_one_position_is_no_group_move():
    options = {"a": {"x": (6, frozenset({1})), "y": (9, frozenset())}}
    calls = []

    def size(state):
        calls.append(state["a"])
        return 50 if state["a"] == "x" else 100

    u2_choose(size, options, {"a": "x"})
    assert calls == ["x", "y"]


def test_a_position_can_drop_its_piece():
    options = {"a": {"x": (6, frozenset()), None: None}}
    assert u2_choose(_toy({}, options), options, {"a": "x"}) == ({"a": None}, 0)


def test_the_search_stops_where_no_move_shrinks_the_game_and_ties_change_nothing():
    options = {"a": {"x": (7, frozenset()), "y": (7, frozenset())}}
    assert u2_choose(_toy({}, options), options, {"a": "x"}) == ({"a": "x"}, 7)


def test_the_best_move_is_taken_first():
    table = {("x", "x"): 100, ("y", "x"): 99, ("x", "y"): 90, ("y", "y"): 101}
    options = {k: {"x": (0, frozenset()), "y": (0, frozenset())} for k in "ab"}
    assert u2_choose(lambda s: table[s["a"], s["b"]], options, dict.fromkeys("ab", "x")) == ({"a": "x", "b": "y"}, 90)


def test_the_game_starts_as_x3_chose_it():
    options, start, size = _game()
    assert size(start) == GAME
    assert (start[_kid(11)], start[_kid(12)], start[_key(0)], start[_key(2)]) == ("x3", "x3", "x2", "x2")


def test_a_kept_rebuild_starts_as_g4_s_and_the_rest_as_g3_s():
    spec = {**JUDGED, "x3_held": [11]}
    x3_spec = {**X3_JUDGED}
    options, start, size = _game(x3=(_x3_arm(bits=(290, 50)), x3_spec), spec={**spec, "control_bits": GAME + 5})
    assert (start[_kid(11)], start[_kid(12)], size(start)) == ("x3", "g4", GAME + 5)
    options, start, _ = _game(x3=(_x3_arm(bits=(310, 50)), x3_spec),
                              spec={**JUDGED, "x3_held": [], "control_bits": X2_GAME})
    assert (start[_kid(11)], start[_kid(12)]) == ("g3", "g4")


def test_a_rebuild_z1_kept_starts_as_z1_s_and_the_other_build_as_v2_s():
    x2_spec = {**X2_JUDGED, "z_held": [0], "control_bits": Z_CONTROL + 10}
    x3_spec = {**X3_JUDGED, "z_held": [0], "x2_held": [], "control_bits": Z_CONTROL + 10}
    _, start, size = _game(x3=(_x3_arm(bits=(310, 50)), x3_spec), x2=(_x2_arm(bits=(60, 200)), x2_spec),
                           spec={**JUDGED, "x3_held": [], "control_bits": Z_CONTROL + 10})
    assert (start[_key(0)], start[_key(2)], size(start)) == ("z", 1, Z_CONTROL + 10)


def test_a_mirror_run_s_piece_plays_its_map_whichever_way_it_reads():
    assert _game(x3=_x3(_x3_arm(leaves=((N + 2,), (2,)))))[0][_kid(11)]["x3"] == (290, frozenset({2}))
    assert _game(x2=_x2(_x2_arm(leaves=((N + 2,), ()))))[0][_key(0)]["x2"] == (40, frozenset({2}))


def test_an_earlier_build_s_own_map_is_no_library_map():
    rows = [_row(0, 60, leaves=(len(LIB) + 1,), found=BUILT[0]), _row(2, 200, found=BUILT[2])]
    assert _game(earlier=_earlier(rows))[0][_key(0)][0] == (60, frozenset())


def test_a_g4_rebuild_s_own_map_is_no_library_map():
    rows = [_row(10, 90, library=G4_LIBRARY), _row(11, 200, library=G4_LIBRARY),
            _row(12, 45, leaves=(2, len(G4_LIBRARY) + 1), library=G4_LIBRARY, found=FOUND)]
    assert _game(g4=_g4(rows))[0][_kid(12)]["g4"] == (45, frozenset({2}))


def test_every_certified_piece_is_an_option_at_its_position():
    options, _, _ = _game()
    assert set(options[_kid(12)]) == {"g3", "g4", "x3", "x4"} and set(options[_kid(11)]) == {"g3", "g4", "x3"}
    assert set(options[_key(0)]) == {0, "z", "x2", None} and set(options[_key(2)]) == {1, "z", "x2", None}


def test_an_unfinished_piece_is_no_option():
    options, _, _ = _game(x4=_x4(complete=False))
    assert "x4" not in options[_kid(12)]
    run, spec = _g4()
    run["roots"][1] = {**run["roots"][1], "complete": False}
    assert "g4" not in _game(g4=(run, spec))[0][_kid(11)]


def test_an_unfinished_rebuild_of_x2_s_is_no_option():
    spec = {**X3_JUDGED, "x2_held": [0], "control_bits": Z_CONTROL - 10}
    game = _game(x3=(_x3_arm(), spec), x2=_x2(_x2_arm(bits=(40, 190), incomplete=(2,))),
                 spec={**JUDGED, "control_bits": Z_CONTROL - 25})
    assert "x2" not in game[0][_key(2)] and "x2" in game[0][_key(0)]


def test_the_whole_game_comes_out_of_the_better_start(monkeypatch):
    best = _size()["bits"]
    real = floor_u2.u2_choose
    calls = []

    def worse_from(size, options, start):
        calls.append(start)
        state, bits = real(size, options, start)
        return (state, bits + 50) if len(calls) == bad else (state, bits)

    monkeypatch.setattr(floor_u2, "u2_choose", worse_from)
    results = []
    for bad in (1, 2):
        calls.clear()
        results.append(_size())
    assert [r["bits"] for r in results] == [best, best] and len(calls) == 2


def test_the_search_finds_the_smaller_pieces_g4_held_back():
    r = _size()
    assert (r["bits"], r["control_bits"], r["ratio"], r["changed"]) == (GAME - 90, GAME, (GAME - 90) / GAME, 1)
    assert r["state"][_kid(11)] == "g4" and r["verdict"] == "supported"


def test_the_bars_are_inclusive():
    bits = _size()["bits"]
    assert _size(spec={**JUDGED, "support": (bits, GAME)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, GAME), "refute": (9, 1)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits, GAME)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits + 1, GAME)})["verdict"] == "inconclusive"


def test_a_game_other_than_x3_s_is_not_run(monkeypatch):
    assert _size(spec={**JUDGED, "control_bits": GAME + 1})["verdict"] == "not_run"
    assert _size(spec={**JUDGED, "x3_held": [11]})["verdict"] == "not_run"
    assert _size(x3=_x3(_x3_arm(bits=(290, 45))))["verdict"] == "not_run"
    monkeypatch.setattr(floor_u2, "x3_size", lambda *a: {"verdict": "supported", "bits": GAME + 1})
    assert _size()["verdict"] == "not_run"


def test_a_control_run_off_its_registration_or_reading_mirrored_is_not_run():
    run, spec = _x4()
    assert _size(x4=({**run, "measurement_fingerprint": "other"}, spec))["verdict"] == "not_run"
    assert _size(x4=_x4(leaves=(2,)))["verdict"] == "not_run"
    assert _size(x4=_x4(library=LIBRARY, leaves=()))["verdict"] == "not_run"


def test_a_g4_rebuild_off_its_library_is_not_run():
    run, spec = _g4()
    run["roots"][1] = _row(11, 200, library=LIBRARY)
    assert _size(g4=(run, spec))["verdict"] == "not_run"
