"""Direct tests for harness/floor_v1.py — V1, every winning move in batches: at the heavy ply-6 positions with a
choice, each winning move no earlier run's root played, built with that move at the root, and the whole game
re-assembled (harness.floor_j1) against the assembly of the earlier runs alone."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_v1 as floor_v1
from harness.floor_v1 import SPEC, v1_report, v1_selected, v1_size
from harness.floor_w6 import w6_report

LIB = [({"3": 0}, 42), ({"5": 1}, 42)]
POSITIONS = {1: ([1, 3], 2000), 2: ([0, 2, 5], 1500), 3: ([4], 5000), 4: ([1, 2], 1499), 5: ([2, 6], 3000)}
EARLIER = [(1, 1, 2000), (2, 0, 2000), (4, 1, 2000), (5, 2, 500), (5, 6, 2000)]
MOVES = [[1, 3], [2, 2], [2, 5]]


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


def _key(index):
    return (index,) + (0,) * 41, 0


def _row(index, move, bits, leaf=0, complete=True, root_move=None, library=LIB):
    board = list(_key(index)[0])
    return {"index": index, "root_move": root_move, "complete": complete, "bits": {"nodes": bits},
            "board": board, "to_move": 0,
            "nodes": [[board, 0, {"move": move}], [[index, 1] + [0] * 40, 0, {"leaf": leaf}]],
            "maps": [{}, *[m for m, _ in library]], "map_empty": [0, *[e for _, e in library]],
            "shared": len(library), "checked": True if complete else None,
            "moves_win": True if complete else None, "leaves_certified": True if complete else None}


def _run(roots, spec):
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"frontier": 148}, "roots": roots}


EARLIER_SPEC = {**SPEC, "roots": len(EARLIER)}


def _earlier(rows=None):
    return [(_run(rows or [_row(i, m, b) for i, m, b in EARLIER], EARLIER_SPEC), EARLIER_SPEC)]


def _scope(**over):
    return {"positions_fp": SPEC["positions_fp"], "library_from": SPEC["library_from"],
            "positions": [{"index": i, "winning": w, "g3_bits": g} for i, (w, g) in POSITIONS.items()], **over}


ASSEMBLY = {"opening": {"moves": 1, "trivial": 0}, "below": {_key(i): [("kid", i)] for i in POSITIONS},
            "base": {("kid", i): (1000, frozenset()) for i in POSITIONS}, "map_cost": {1: 100, 2: 200},
            "library": LIB}
CONTROL = 4 + 4 * 4 + 4 * 1000 + 500
JUDGED = {**SPEC, "roots": 3, "indices": [1, 2], "root_moves": MOVES, "j1_bits": CONTROL}


def _arm(bits=(300, 900, 700), leaves=(0, 0, 0), incomplete=(), moves=MOVES):
    return _run([_row(i, m, b, leaf, complete=k not in incomplete, root_move=m)
                 for k, ((i, m), b, leaf) in enumerate(zip(moves, bits, leaves))], JUDGED)


def _size(arm=None, earlier=None, spec=JUDGED, scope=None, assembly=ASSEMBLY):
    return v1_size(arm or _arm(), earlier or _earlier(), scope or _scope(), assembly, spec)


def test_the_spec_names_the_moves_its_selection_finds():
    assert SPEC["roots"] == len(SPEC["root_moves"])
    assert SPEC["indices"] == sorted({i for i, _ in SPEC["root_moves"]})
    assert v1_report is w6_report


def test_the_selection_is_every_unbuilt_winning_move_at_heavy_positions_with_a_choice():
    rows = [r for e, _ in _earlier() for r in e["roots"]]
    assert v1_selected(_scope(), rows, JUDGED) == MOVES


def test_a_root_the_builder_left_as_a_leaf_builds_no_move():
    rows = [r for e, _ in _earlier() for r in e["roots"]]
    leaf = {**_row(1, 3, 10), "nodes": [[list(_key(1)[0]), 0, {"leaf": 1}]]}
    assert v1_selected(_scope(), rows + [leaf], JUDGED) == MOVES


def test_the_selection_needs_the_registered_positions_and_library():
    assert v1_selected(_scope(positions_fp="other"), [], JUDGED) is None
    assert v1_selected(_scope(library_from="other"), [], JUDGED) is None


def test_the_whole_game_takes_the_cheapest_builds_and_is_judged_against_the_earlier_assembly():
    r = _size()
    assert r == {"verdict": "supported", "bits": CONTROL - 704 - 304, "control_bits": CONTROL,
                 "ratio": (CONTROL - 1008) / CONTROL, "taken": [[1, 3], [2, 5]], "unfinished": False}


def test_a_build_s_new_library_maps_are_paid_once():
    r = _size(arm=_arm(bits=(300, 900, 700), leaves=(1, 0, 1)))
    assert r["bits"] == CONTROL - 1008 + 100 and r["taken"] == [[1, 3], [2, 5]]


def test_no_saving_once_every_build_finished_refutes():
    r = _size(arm=_arm(bits=(5000, 5000, 5000)))
    assert r["verdict"] == "refuted" and r["bits"] == CONTROL and r["taken"] == []


def test_an_unfinished_build_is_never_taken_and_cannot_refute():
    r = _size(arm=_arm(bits=(5000, 5000, 1), incomplete=(2,)))
    assert r["verdict"] == "inconclusive" and r["bits"] == CONTROL and r["unfinished"]


def test_the_bars_are_inclusive():
    bits = CONTROL - 1008
    assert _size(spec={**JUDGED, "support": (bits, CONTROL)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, CONTROL)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 2), "refute": (bits, CONTROL)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 2), "refute": (bits + 1, CONTROL)})["verdict"] == "inconclusive"


def test_the_game_is_never_worse_than_the_earlier_assembly(monkeypatch):
    real = floor_v1.j1_choose

    def worse(opening, below, base, candidates, map_cost):
        chosen, bits = real(opening, below, base, candidates, map_cost)
        if sum(map(len, candidates.values())) == len(EARLIER):
            return chosen, bits
        return {k: options[-1][0] for k, options in candidates.items() if options[-1][0][0] == 1}, CONTROL + 9

    monkeypatch.setattr(floor_v1, "j1_choose", worse)
    r = _size()
    assert r["bits"] == CONTROL and r["taken"] == [] and r["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_run():
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"})["verdict"] == "not_run"
    earlier = _earlier()
    earlier[0][0]["builder"] = {"other": 1}
    assert _size(earlier=earlier)["verdict"] == "not_run"


def test_moves_other_than_the_selection_s_are_not_run():
    assert _size(spec={**JUDGED, "root_moves": MOVES[:2]})["verdict"] == "not_run"
    two = {**JUDGED, "roots": 2, "root_moves": MOVES[:2]}
    assert _size(arm=_arm(moves=MOVES[:2]), spec=two)["verdict"] == "not_run"
    assert _size(arm=_arm(moves=[[1, 3], [2, 2], [2, 2]]))["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][1]["nodes"][0][2]["move"] = 5
    assert _size(arm=arm)["verdict"] == "not_run"


def test_pieces_off_the_library_are_not_run():
    other = [LIB[1], LIB[0]]
    assert _size(arm=_run([_row(1, 3, 300, root_move=3), _row(2, 2, 900, root_move=2),
                           _row(2, 5, 700, root_move=5, library=other)], JUDGED))["verdict"] == "not_run"
    rows = [_row(i, m, b) for i, m, b in EARLIER]
    rows[0] = _row(1, 1, 2000, library=other)
    assert _size(earlier=_earlier(rows))["verdict"] == "not_run"


def test_an_unfinished_build_off_the_library_is_only_left_out():
    other = [LIB[1], LIB[0]]
    arm = _run([_row(1, 3, 300, root_move=3), _row(2, 2, 900, root_move=2),
                _row(2, 5, 700, root_move=5, complete=False, library=other)], JUDGED)
    assert _size(arm=arm)["taken"] == [[1, 3], [2, 2]]


def test_an_earlier_assembly_other_than_registered_is_not_run():
    assert _size(spec={**JUDGED, "j1_bits": CONTROL + 1})["verdict"] == "not_run"


def test_a_tie_keeps_the_earlier_build():
    rows = [_row(i, m, 300 if i == 1 else b) for i, m, b in EARLIER]
    r = _size(earlier=_earlier(rows), spec={**JUDGED, "j1_bits": CONTROL - 704})
    assert r["taken"] == [[2, 5]] and r["bits"] == CONTROL - 1008
