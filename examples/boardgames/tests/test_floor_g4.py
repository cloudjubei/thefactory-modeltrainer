"""Direct tests for harness/floor_g4.py — G4, the assembled whole game's library pass: the ply-8 positions the
assembled game still takes from G3, rebuilt from every map it holds (G3's library, then the own maps of those G3
strategies and of the builds it takes), and the whole game re-assembled with them."""
from __future__ import annotations

import pytest

import harness.floor_g4 as floor_g4
import harness.floor_j1 as floor_j1
from harness.floor_g4 import SPEC, g4_library, g4_report, g4_selected, g4_size
from harness.floor_w6 import w6_report

LIB = [({"3": 0}, 42), ({"5": 1}, 42)]
OWN = {10: [[{"0": 0}, 7]], 11: [[{"1": 0}, 7]], 12: [], 13: [[{"2": 0}, 7]]}
BUILT = {0: [[{"4": 0}, 7]], 2: [[{"6": 0}, 7]]}
LIBRARY = LIB + OWN[10] + OWN[11] + BUILT[2]


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


def _key(index):
    return (index,) + (0,) * 41, 0


def _kid(index):
    return (index, 1) + (0,) * 40, 0


def _row(index, bits, leaves=(), library=LIB, found=(), complete=True, at=None, root_move=None):
    board = list((_key if index < 10 else _kid)(index if at is None else at)[0])
    nodes = [[board, 0, {"move": 3}]] + [[[index, 2 + n] + [0] * 40, 0, {"leaf": leaf}]
                                         for n, leaf in enumerate(leaves)]
    return {"index": index, "root_move": root_move, "complete": complete, "bits": {"nodes": bits}, "board": board,
            "to_move": 0, "nodes": nodes, "maps": [{}, *[m for m, _ in library], *[m for m, _ in found]],
            "map_empty": [0, *[e for _, e in library], *[e for _, e in found]], "shared": len(library),
            "found_maps": [list(m) for m in found], "checked": True if complete else None,
            "moves_win": True if complete else None, "leaves_certified": True if complete else None}


def _run(roots, spec):
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"frontier": spec["frontier"]}, "roots": roots}


def _earlier(rows=None):
    rows = rows or [_row(0, 5000, found=BUILT[0]), _row(2, 200, found=BUILT[2])]
    spec = {**SPEC, "roots": len(rows), "frontier": 148}
    return [(_run(rows, spec), spec)]


G3 = {"roots": [_row(i, 0, found=OWN[i]) for i in (10, 11, 12, 13)]}
ASSEMBLY = {"opening": {"moves": 1, "trivial": 0},
            "below": {_key(0): [_kid(10), _kid(11)], _key(1): [_kid(11), _kid(12), _kid(14)], _key(2): [_kid(13)]},
            "base": {_kid(10): (100, frozenset({1})), _kid(11): (300, frozenset()), _kid(12): (50, frozenset({2})),
                     _kid(13): (1000, frozenset())},
            "map_cost": {1: 40, 2: 60}, "library": LIB}
CONTROL = 4 + 2 * 4 + 100 + 300 + 50 + 1 + 40 + 60 + 200
JUDGED = {**SPEC, "roots": 3, "indices": [10, 11, 12], "taken": [[0, 2, None]], "control_bits": CONTROL}


def _cost(monkeypatch, costs=(40, 60, 11, 12, 13)):
    monkeypatch.setattr("harness.floor_g4.map_bits", lambda levels, empty, level_bits: costs[
        [{int(c): k for c, k in m.items()} for m, _ in LIBRARY].index(levels)])


def _arm(bits=(90, 200, 50), leaves=((1,), (5,), (2,)), incomplete=()):
    return _run([_row(i, b, leaves=l, library=LIBRARY, complete=n not in incomplete)
                 for n, (i, b, l) in enumerate(zip((10, 11, 12), bits, leaves))], JUDGED)


def _size(arm=None, earlier=None, spec=JUDGED, g3=G3, assembly=ASSEMBLY):
    return g4_size(arm or _arm(), earlier or _earlier(), g3, assembly, spec)


def test_the_spec_names_its_positions_and_the_builds_it_takes():
    assert SPEC["roots"] == len(SPEC["indices"]) == len(set(SPEC["indices"]))
    assert SPEC["indices"] == sorted(SPEC["indices"])
    assert SPEC["builder"]["seconds"] == 7200.0 and SPEC["frontier"] == 671
    assert g4_report is w6_report


def test_the_positions_are_the_g3_strategies_the_assembled_game_keeps():
    assert g4_selected(_earlier(), G3, ASSEMBLY) == ([10, 11, 12], [[0, 2, None]])


def test_the_library_is_g3_s_then_the_kept_strategies_own_maps_then_the_taken_builds():
    earlier = [run for run, _ in _earlier()]
    assert g4_library(LIB, G3, earlier, JUDGED) == LIBRARY
    assert g4_library(LIB, G3, earlier, {**JUDGED, "indices": [11, 13], "taken": []}) == LIB + OWN[11] + OWN[13]


def test_the_rebuilt_strategies_and_their_shared_maps_are_paid_once(monkeypatch):
    _cost(monkeypatch)
    r = _size()
    bits = 4 + 8 + 90 + 200 + 50 + 1 + 40 + 60 + 13 + 200
    assert r == {"verdict": "supported", "bits": bits, "control_bits": CONTROL, "ratio": bits / CONTROL,
                 "rebuilt": 3, "unfinished": False}


def test_a_rebuilt_strategy_s_new_map_paid_once_however_many_play_it(monkeypatch):
    _cost(monkeypatch)
    r = _size(arm=_arm(leaves=((5,), (5,), (2,))))
    assert r["bits"] == 4 + 8 + 90 + 200 + 50 + 1 + 60 + 13 + 200


def test_a_rebuild_that_grows_the_game_keeps_g3_s_strategy(monkeypatch):
    _cost(monkeypatch)
    r = _size(arm=_arm(bits=(90, 290, 50)))
    assert r["bits"] == 4 + 8 + 90 + 300 + 50 + 1 + 40 + 60 + 200 and r["rebuilt"] == 2
    r = _size(arm=_arm(bits=(90, 287, 50)))
    assert r["bits"] == 4 + 8 + 90 + 287 + 50 + 1 + 40 + 60 + 13 + 200 and r["rebuilt"] == 3


def test_every_rebuild_that_alone_grows_the_game_goes_back(monkeypatch):
    _cost(monkeypatch)
    r = _size(arm=_arm(bits=(190, 290, 20)))
    assert r["bits"] == 4 + 8 + 100 + 300 + 20 + 1 + 40 + 60 + 200 and r["rebuilt"] == 1


def test_the_ply_6_builds_are_chosen_again_with_the_rebuilds(monkeypatch):
    _cost(monkeypatch)
    assembly = {**ASSEMBLY, "base": {**ASSEMBLY["base"], _kid(13): (230, frozenset())}}
    rows = [_row(0, 5000, found=BUILT[0]), _row(2, 200, leaves=(2,), found=BUILT[2])]
    r = _size(arm=_arm(leaves=((1,), (5,), ())), earlier=_earlier(rows), assembly=assembly)
    assert r["bits"] == 4 + 12 + 90 + 200 + 50 + 230 + 1 + 40 + 13 and r["rebuilt"] == 3


def test_only_rebuilds_below_positions_keeping_g3_s_pieces_count(monkeypatch):
    _cost(monkeypatch)
    real = floor_g4.v2_assemble

    def last_takes_0(rows, a):
        chosen, bits = real(rows, a)
        return ({**chosen, _key(0): 0}, bits) if a is not ASSEMBLY else (chosen, bits)

    monkeypatch.setattr(floor_g4, "v2_assemble", last_takes_0)
    assert _size()["rebuilt"] == 2


def test_the_game_is_never_worse_than_the_assembled_one(monkeypatch):
    _cost(monkeypatch)
    r = _size(arm=_arm(bits=(900, 900, 900)))
    assert r["verdict"] == "refuted" and r["bits"] == CONTROL and r["rebuilt"] == 0
    real = floor_g4.v2_assemble
    monkeypatch.setattr(floor_g4, "v2_assemble", lambda rows, a: real(rows, a) if a is ASSEMBLY else ({}, CONTROL + 9))
    r = _size()
    assert r["verdict"] == "refuted" and r["bits"] == CONTROL and r["rebuilt"] == 0


def test_an_unfinished_rebuild_keeps_g3_s_strategy_and_cannot_refute(monkeypatch):
    _cost(monkeypatch)
    r = _size(arm=_arm(bits=(900, 900, 1), incomplete=(2,)))
    assert r["verdict"] == "inconclusive" and r["bits"] == CONTROL and r["unfinished"] and r["rebuilt"] == 0


def test_the_bars_are_inclusive(monkeypatch):
    _cost(monkeypatch)
    bits = 4 + 8 + 90 + 200 + 50 + 1 + 40 + 60 + 13 + 200
    assert _size(spec={**JUDGED, "support": (bits, CONTROL)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, CONTROL)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits, CONTROL)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits + 1, CONTROL)})["verdict"] == "inconclusive"


def test_a_run_off_its_registration_is_not_run(monkeypatch):
    _cost(monkeypatch)
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"})["verdict"] == "not_run"
    earlier = _earlier()
    earlier[0][0]["builder"] = {"other": 1}
    assert _size(earlier=earlier)["verdict"] == "not_run"
    assert _size(spec={**JUDGED, "control_bits": CONTROL + 1})["verdict"] == "not_run"


def test_positions_or_builds_other_than_the_selection_s_are_not_run(monkeypatch):
    _cost(monkeypatch)
    assert _size(spec={**JUDGED, "roots": 2, "indices": [10, 11]},
                 arm=_run(_arm()["roots"][:2], {**JUDGED, "roots": 2}))["verdict"] == "not_run"
    none = LIB + OWN[10] + OWN[11]
    arm = _run([_row(i, b, leaves=l, library=none) for i, b, l in ((10, 90, (1,)), (11, 200, ()), (12, 50, (2,)))],
               JUDGED)
    assert _size(arm=arm, spec={**JUDGED, "taken": []})["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][2] = _row(13, 50, leaves=(2,), library=LIBRARY)
    assert _size(arm=arm)["verdict"] == "not_run"


def test_a_row_at_another_position_than_its_index_is_not_run(monkeypatch):
    _cost(monkeypatch)
    arm = _arm()
    arm["roots"][1] = _row(11, 280, leaves=(5,), library=LIBRARY, at=13)
    assert _size(arm=arm)["verdict"] == "not_run"
    rows = [_row(0, 5000, found=BUILT[0]), _row(2, 200, found=BUILT[2])]
    rows[1]["index"] = 1
    assert _size(earlier=_earlier(rows), spec={**JUDGED, "taken": [[0, 1, None]]})["verdict"] == "not_run"


def test_rebuilds_off_the_library_are_not_run_unless_unfinished(monkeypatch):
    _cost(monkeypatch)
    arm = _arm()
    arm["roots"][0] = _row(10, 90, leaves=(1,), library=LIBRARY[:4])
    assert _size(arm=arm)["verdict"] == "not_run"
    arm["roots"][0] = _row(10, 90, leaves=(1,), library=LIBRARY[:4], complete=False)
    assert _size(arm=arm)["rebuilt"] == 2
    rows = [_row(0, 5000, library=LIB[:1]), _row(2, 200, found=BUILT[2])]
    assert _size(earlier=_earlier(rows))["verdict"] == "not_run"
