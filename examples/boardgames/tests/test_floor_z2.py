"""Direct tests for harness/floor_z2.py — Z2, the ply-6 builds of U2's game rebuilt with both readings from every map
the game holds: X2's library, then the own maps of the pieces the game plays, judged on the whole game with every
rebuild one more option for U2's search."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_u2 as floor_u2
import harness.floor_x2 as floor_x2
import harness.floor_x3 as floor_x3
import harness.floor_z2 as floor_z2
from harness.floor_u2 import SPEC as U2
from harness.floor_w6 import w6_report
from harness.floor_x2 import SPEC as X2
from harness.floor_z1 import LIBRARY_SOURCES as Z1_SOURCES
from harness.floor_z2 import LIBRARY_SOURCES, SPEC, _taken, z2_library, z2_library_from, z2_report, z2_size
from tests.test_floor_g4 import ASSEMBLY, G3, _key, _kid, _row, _run
from tests.test_floor_u2 import GAME, JUDGED as U2_JUDGED, _x3, _x4
from tests.test_floor_x2 import LIBRARY, N, READS, _z
from tests.test_floor_x3 import _arm as _x3_arm, _x2
from tests.test_floor_z1 import _earlier, _g4

NEW = [[{"6": 3}, 7]]
GROWN = [*LIBRARY, *NEW]
GROWN_READS = [*GROWN, *GROWN]
CONTROL = GAME - 90
NO_OWN = {name: [] for name in SPEC["own"]}
JUDGED = {**SPEC, "roots": 2, "indices": [0, 2], "root_moves": [[0, None], [2, None]], "own": NO_OWN,
          "control_bits": CONTROL}
OWN_X3 = {**JUDGED, "own": {**NO_OWN, "x3": [12]}}


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


@pytest.fixture(autouse=True)
def _costs(monkeypatch):
    costs = (40, 60, 11, 12, 13, 14, 15, 17)

    def cost(levels, empty, level_bits):
        return costs[[{int(c): k for c, k in m.items()} for m, _ in GROWN].index(levels)]

    for module in (floor_x2, floor_x3, floor_u2, floor_z2):
        monkeypatch.setattr(module, "map_bits", cost)


def _arm(bits=(30, 185), leaves=((), ()), incomplete=(), library=READS, root_moves=(None, None), spec=JUDGED):
    return _run([_row(i, b, leaves=l, library=library, complete=i not in incomplete, root_move=m)
                 for i, b, l, m in zip((0, 2), bits, leaves, root_moves)], spec)


def _x3_with_own(bits=40):
    arm = _x3_arm()
    arm["roots"][1] = _row(12, bits, leaves=(2,), library=READS, found=NEW)
    return _x3(arm)


def _size(arm=None, x3=None, x4=None, spec=JUDGED, u2=U2_JUDGED):
    return z2_size(arm or _arm(), x4 or _x4(), x3 or _x3(), _x2(), _z(), _earlier(), _g4(), G3, ASSEMBLY, u2, spec)


def test_the_spec_rebuilds_u2_s_builds_with_x2_s_builder():
    assert SPEC["builder"] == X2["builder"] and SPEC["measurement_fp"] == X2["measurement_fp"]
    assert SPEC["frontier"] == 148 and SPEC["builder"]["seconds"] == 21_600.0 and SPEC["builder"]["mirror"]
    assert SPEC["roots"] == 50 == len(SPEC["root_moves"]) == len({i for i, _ in SPEC["root_moves"]})
    assert SPEC["indices"] == sorted(i for i, _ in SPEC["root_moves"])
    assert list(SPEC["own"]) == ["earlier", "g3", "g4", "z", "x2", "x3", "x4"]
    assert all(labels == sorted(labels) for labels in SPEC["own"].values())
    assert SPEC["library_from"] == "c49_z2_library.json.gz" and z2_report is w6_report
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (47, 39)
    assert (SPEC["support"], SPEC["refute"]) == ((99, 100), (999, 1000)) and SPEC["control_bits"] == 165_023
    assert "kept" not in SPEC and "z_held" not in SPEC
    assert LIBRARY_SOURCES == (*Z1_SOURCES, "c49_z1.json.gz", "c49_x2.json.gz", "c49_x3.json.gz", "c49_x4.json.gz")


def test_the_library_is_x2_s_then_the_own_maps_it_does_not_hold():
    runs = {"a": {1: {"found_maps": [[{"1": 1}, 7], [{"2": 2}, 7]]}}, "b": {"x": {"found_maps": [[{"3": 3}, 7]]}}}
    held = [({"2": 2}, 7)]
    spec = {"own": {"b": ["x"], "a": [1]}}
    assert z2_library(held, runs, spec) == [[{"2": 2}, 7], [{"3": 3}, 7], [{"1": 1}, 7]]


def test_two_pieces_holding_one_map_add_it_once():
    runs = {"a": {1: {"found_maps": [[{"1": 1}, 7]]}, 2: {"found_maps": [[{"1": 1}, 7]]}}}
    assert z2_library([], runs, {"own": {"a": [1, 2]}}) == [[{"1": 1}, 7]]


def test_a_map_with_another_empty_value_is_another_map():
    runs = {"a": {1: {"found_maps": [[{"1": 1}, 8]]}}}
    assert z2_library([({"1": 1}, 7)], runs, {"own": {"a": [1]}}) == [[{"1": 1}, 7], [{"1": 1}, 8]]


def test_the_library_file_comes_twice_from_the_sources(monkeypatch):
    def row(i, found):
        return {"index": i, "found_maps": [found]}

    runs = [{"roots": [row(i, [{str(i): i}, 7])]} for i in range(1, 8)]
    earlier, g3, g4, z, x2, x3, x4 = runs
    seen = []
    monkeypatch.setattr(floor_z2, "z1_library_from", lambda *a: seen.append(a) or [[{"0": 0}, 7]])
    monkeypatch.setattr(floor_z2, "SPEC", {"own": {"earlier": [0], "g3": [2], "g4": [3], "z": [4], "x2": [5], "x3": [6],
                                                   "x4": [7]}})
    grown = [[{"0": 0}, 7], *[[{str(i): i}, 7] for i in range(1, 8)]]
    assert z2_library_from("g2", "f2", g3, earlier, g4, z, x2, x3, x4) == [*grown, *grown]
    assert seen == [("g2", "f2", g3, earlier, g4)]


def test_a_smaller_rebuild_replaces_its_build():
    r = _size()
    assert {k: v for k, v in r.items() if k != "state"} == {
        "verdict": "supported", "bits": CONTROL - 10, "control_bits": CONTROL, "ratio": (CONTROL - 10) / CONTROL,
        "rebuilt": 1, "unfinished": False}
    assert (r["state"][_key(0)], r["state"][_key(2)]) == ("z2", "x2")


def test_a_larger_rebuild_leaves_u2_s_game():
    r = _size(arm=_arm(bits=(50, 190)))
    assert (r["verdict"], r["bits"], r["rebuilt"]) == ("refuted", CONTROL, 0)


def test_a_rebuild_playing_a_grown_map_pays_it_once_however_it_reads():
    x3 = _x3_with_own()
    n = N + 1
    arm = _arm(bits=(20, 175), leaves=((n + 8,), (8,)), library=GROWN_READS, spec=OWN_X3)
    r = _size(arm=arm, x3=x3, spec=OWN_X3)
    assert r["bits"] == CONTROL - 30 + 17 and r["rebuilt"] == 2


def test_a_grown_map_that_costs_more_than_it_saves_stays_out():
    arm = _arm(bits=(35, 185), leaves=((8,), ()), library=GROWN_READS, spec=OWN_X3)
    assert _size(arm=arm, x3=_x3_with_own(), spec=OWN_X3)["bits"] == CONTROL


def test_the_game_is_never_worse_than_u2_s(monkeypatch):
    monkeypatch.setattr(floor_z2, "u2_choose", lambda size, options, start: (start, size(start)))
    assert _size()["bits"] <= CONTROL


def test_the_search_runs_from_u2_s_game_and_from_every_position_s_smallest_piece(monkeypatch):
    calls, real = [], floor_z2.u2_choose
    monkeypatch.setattr(floor_z2, "u2_choose", lambda size, options, start: calls.append(start)
                        or real(size, options, start))
    state = _size()["state"]
    game, smallest = calls[-2:]
    assert (game[_key(0)], game[_key(2)]) == ("x2", "x2")
    assert (smallest[_key(0)], smallest[_key(2)], smallest[_kid(11)]) == ("z2", "x2", "g4")
    assert state[_key(0)] == "z2"


def test_the_smaller_of_the_two_starts_is_kept(monkeypatch):
    real = floor_z2.u2_choose
    best = _size()["bits"]
    for bad in (1, 2):
        calls = []

        def worse(size, options, start):
            calls.append(start)
            state, bits = real(size, options, start)
            return (state, bits + 50) if len(calls) == bad else (state, bits)

        monkeypatch.setattr(floor_z2, "u2_choose", worse)
        assert _size()["bits"] == best
        monkeypatch.setattr(floor_z2, "u2_choose", real)


def test_an_unfinished_rebuild_is_no_option_and_cannot_refute():
    r = _size(arm=_arm(bits=(900, 900), incomplete=(0,)))
    assert (r["verdict"], r["bits"], r["unfinished"], r["rebuilt"]) == ("inconclusive", CONTROL, True, 0)
    assert _size(arm=_arm(bits=(30, 185), incomplete=(2,)))["bits"] == CONTROL - 10


def test_the_bars_are_inclusive():
    bits = CONTROL - 10
    assert _size(spec={**JUDGED, "support": (bits, CONTROL)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, CONTROL)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits, CONTROL)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits + 1, CONTROL)})["verdict"] == "inconclusive"


def test_a_run_off_its_registration_is_not_run():
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"}) == {"verdict": "not_run"}


def test_a_game_other_than_u2_s_is_not_run():
    assert _size(spec={**JUDGED, "control_bits": CONTROL + 1}) == {"verdict": "not_run"}
    assert _size(u2={**U2_JUDGED, "control_bits": GAME + 1}) == {"verdict": "not_run"}


def test_a_game_whose_size_differs_from_u2_s_is_not_run(monkeypatch):
    real = floor_z2.j1_bits
    monkeypatch.setattr(floor_z2, "j1_bits", lambda *a: {"bits": real(*a)["bits"] + 1})
    assert _size() == {"verdict": "not_run"}


def test_builds_other_than_the_game_takes_are_not_run():
    spec = {**JUDGED, "roots": 1, "indices": [2], "root_moves": [[2, None]]}
    assert _size(arm=_run(_arm()["roots"][1:], spec), spec=spec) == {"verdict": "not_run"}


def test_builds_the_game_takes_on_another_root_move_are_not_run():
    spec = {**JUDGED, "root_moves": [[0, 3], [2, None]]}
    assert _size(arm=_arm(root_moves=(3, None)), spec=spec) == {"verdict": "not_run"}


def test_a_rebuild_off_its_root_move_is_not_run():
    arm = _arm()
    arm["roots"][0]["root_move"] = 3
    assert _size(arm=arm) == {"verdict": "not_run"}
    arm = _arm(root_moves=(3, None))
    assert _size(arm=arm, spec={**JUDGED, "root_moves": [[0, 3], [2, None]]}) == {"verdict": "not_run"}


def test_a_rebuild_playing_another_move_than_its_root_move_is_not_run(monkeypatch):
    real = floor_z2._taken

    def build_on_move_3(*args):
        return [(n, label, {**r, "root_move": 3} if r["index"] == 0 else r) for n, label, r in real(*args)]

    monkeypatch.setattr(floor_z2, "_taken", build_on_move_3)
    spec = {**JUDGED, "root_moves": [[0, 3], [2, None]]}
    assert _size(arm=_arm(root_moves=(3, None)), spec=spec)["verdict"] != "not_run"
    arm = _arm(root_moves=(3, None))
    arm["roots"][0]["nodes"][0][2]["move"] = 4
    assert _size(arm=arm, spec=spec) == {"verdict": "not_run"}


def test_an_earlier_build_is_taken_by_its_label_and_a_run_s_by_its_name():
    earlier, rebuilt = _row(0, 60), _row(2, 50)
    runs = {"earlier": {0: earlier, 1: _row(2, 70)}, "z": {0: _row(0, 50), 2: rebuilt}}
    below = {_key(0): [], _key(2): []}
    assert _taken({_key(0): 0, _key(2): "z"}, runs, below) == [("earlier", 0, earlier), ("z", 2, rebuilt)]


def test_a_row_at_another_position_than_its_index_is_not_run():
    arm = _arm()
    arm["roots"][1] = _row(2, 185, library=READS, at=1)
    assert _size(arm=arm) == {"verdict": "not_run"}


def test_a_library_off_the_game_s_own_maps_is_not_run():
    assert _size(x3=_x3_with_own()) == {"verdict": "not_run"}
    assert _size(spec=OWN_X3) == {"verdict": "not_run"}


def test_only_pieces_the_game_plays_grow_the_library():
    r = _size()
    assert r["state"][_kid(10)] == "g3" and G3["roots"][0]["found_maps"] and JUDGED["own"]["g3"] == []
    assert r["verdict"] != "not_run"


def test_rows_off_their_library_are_not_run_unless_unfinished():
    assert _size(arm=_arm(library=LIBRARY)) == {"verdict": "not_run"}
    assert _size(arm=_arm(library=LIBRARY, incomplete=(0, 2)))["bits"] == CONTROL
