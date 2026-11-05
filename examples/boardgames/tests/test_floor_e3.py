"""Direct tests for harness/floor_e3.py — E3, every winning move at ply 6: at E1's positions, each winning move but
the one E1's root played, built with that move at the root, and the cheapest complete build at each position judged
against E1's builds for the same positions."""
from __future__ import annotations

import pytest

import harness.floor_e3 as floor_e3
from harness.floor_e1 import SPEC as E1
from harness.floor_e3 import SPEC, e3_report, e3_selected, e3_size
from harness.floor_w6 import w6_report
from harness.steady_state import map_bits

LIB = [({"3": 0}, 42), ({"5": 1}, 42), ({"10": 1}, 42)]
PLAYED = {0: 2, 1: 1, 2: 2, 3: 0, 4: 4, 5: 5, 6: 6, 7: 1}
WINNING = {1: [1, 3], 2: [0, 2, 5]}
MOVES = [[1, 3], [2, 0], [2, 5]]


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_e3, "_coded", lambda r: r["bits"]["nodes"])


def _row(index, move, bits, leaf=1, complete=True, root_move=None, library=LIB):
    board = [index] + [0] * 41
    return {"index": index, "root_move": root_move, "complete": complete, "bits": {"nodes": bits},
            "board": board, "to_move": 0,
            "nodes": [[board, 0, {"move": move}], [[index, 1] + [0] * 40, 0, {"leaf": leaf}]],
            "maps": [{}, *[m for m, _ in library]], "map_empty": [0, *[e for _, e in library]],
            "shared": len(library), "checked": True if complete else None,
            "moves_win": True if complete else None, "leaves_certified": True if complete else None}


def _e(roots, spec):
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"frontier": 148}, "roots": roots}


def _e1(bits=(100, 100), leaves=(1, 1)):
    sized = {1: (bits[0], leaves[0]), 2: (bits[1], leaves[1])}
    return _e([_row(i, m, *sized.get(i, (500, 1))) for i, m in PLAYED.items()], E1)


def _scope(**over):
    return {"positions_fp": SPEC["positions_fp"], "library_from": SPEC["library_from"],
            "positions": [{"index": i, "winning": WINNING.get(i, [m])} for i, m in PLAYED.items()], **over}


def _arm(bits=(100, 100, 100), leaves=(1, 1, 1), incomplete=()):
    return _e([_row(i, m, b, leaf, complete=k not in incomplete, root_move=m)
               for k, ((i, m), b, leaf) in enumerate(zip(MOVES, bits, leaves))], JUDGED)


G3 = {"roots": [_row(9, 0, 50, leaf=1)]}
JUDGED = {**SPEC, "roots": 3, "indices": [1, 2], "root_moves": MOVES, "e1_bits": 200}


def _size(arm=None, e1=None, spec=JUDGED, scope=None, library=LIB):
    return e3_size(arm or _arm(), e1 or _e1(), G3, library, scope or _scope(), spec)


def test_e3_builds_every_other_winning_move_at_e1_s_positions():
    assert SPEC["indices"] == [18, 69, 79, 115, 130, 145, 146] and SPEC["roots"] == 13
    assert SPEC["root_moves"] == [[18, 4], [69, 1], [69, 6], [79, 1], [79, 4], [79, 6], [115, 2], [130, 2], [145, 0],
                                  [145, 1], [145, 4], [146, 4], [146, 5]]
    assert (SPEC["complete_support"], SPEC["complete_refute"], SPEC["support"], SPEC["refute"]) == (
        12, 10, (9, 10), (97, 100))
    assert SPEC["e1_bits"] == 28_712 and SPEC["measurement_fp"] == "3981ea9f5020"
    assert SPEC["scope"] == "c49_o1_scope.json.gz"
    changed = {"measurement_fp", "roots", "indices", "root_moves", "scope", "complete_support", "complete_refute",
               "support", "refute", "e1_bits", "g3_subtree_bits"}
    assert {k: v for k, v in SPEC.items() if k not in changed} == {k: v for k, v in E1.items() if k not in changed}
    assert "g3_subtree_bits" in SPEC and e3_report is w6_report


def test_the_moves_are_each_winning_move_but_the_one_e1_s_root_played():
    assert e3_selected(_scope(), _e1(), JUDGED) == MOVES
    assert e3_selected(_scope(positions_fp="x"), _e1(), JUDGED) is None
    assert e3_selected(_scope(library_from="other.json.gz"), _e1(), JUDGED) is None


def test_the_cheapest_complete_build_at_each_position_is_judged_and_a_tie_keeps_e1_s():
    r = _size(_arm(bits=(60, 100, 80)))
    assert (r["bits"], r["control_bits"], r["moves"], r["verdict"]) == (140, 200, {1: 3, 2: 5}, "supported")
    r = _size(_arm(bits=(100, 120, 100)))
    assert (r["bits"], r["moves"], r["verdict"]) == (200, {1: 1, 2: 2}, "refuted")


def test_the_verdict_is_judged_against_e1_s_bits_for_the_same_positions():
    for best, verdict in ((80, "supported"), (81, "inconclusive"), (93, "inconclusive"), (94, "refuted")):
        r = _size(_arm(bits=(best, 500, 500)))
        assert (r["bits"], r["ratio"], r["verdict"]) == (best + 100, (best + 100) / 200, verdict)


def test_an_unfinished_build_is_never_chosen_and_keeps_the_verdict_from_refuting():
    r = _size(_arm(bits=(10, 500, 500), incomplete=(0,)))
    assert (r["bits"], r["moves"], r["unfinished"], r["verdict"]) == (200, {1: 1, 2: 2}, True, "inconclusive")
    r = _size(_arm(bits=(500, 10, 500), incomplete=(0,)))
    assert (r["bits"], r["moves"], r["verdict"]) == (110, {1: 1, 2: 0}, "supported")


def test_library_maps_g3_never_played_are_charged_once_to_the_builds_that_play_them():
    two, three = (map_bits({int(c): k for c, k in LIB[i][0].items()}, 42, 3) for i in (1, 2))
    spec = {**JUDGED, "e1_bits": 200 + two}
    r = _size(_arm(bits=(10, 10, 500), leaves=(3, 2, 1)), e1=_e1(leaves=(2, 1)), spec=spec)
    assert r["control_bits"] == 200 + two
    assert r["bits"] == min(10 + three, 100 + two) + 10 + two
    assert _size(e1=_e1(leaves=(2, 1)))["verdict"] == "not_run"
    r = _size(_arm(bits=(100 - three + 1, 500, 500), leaves=(3, 1, 1)))
    assert (r["moves"], r["bits"]) == ({1: 1, 2: 2}, 200)


def test_builds_off_the_registration_are_not_judged():
    assert _size(scope=_scope(positions_fp="x"))["verdict"] == "not_run"
    assert _size(spec={**JUDGED, "root_moves": MOVES[:2]})["verdict"] == "not_run"
    assert _size(spec={**JUDGED, "e1_bits": 201})["verdict"] == "not_run"
    assert _size(library=LIB[::-1])["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][2] = _row(2, 0, 100, root_move=0)
    assert _size(arm)["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][1]["root_move"] = 5
    assert _size(arm)["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][2]["nodes"][0][2]["move"] = 0
    assert _size(arm)["verdict"] == "not_run"
    assert _size(_e(_arm()["roots"], {**JUDGED, "measurement_fp": "other"}))["verdict"] == "not_run"
    e1 = _e1()
    e1["roots"][0]["complete"] = False
    assert _size(e1=e1)["verdict"] == "not_run"
    e1 = _e1()
    e1["measurement_fingerprint"] = "other"
    assert _size(e1=e1)["verdict"] == "not_run"
    e1 = _e1()
    e1["roots"][1] = _row(1, 1, 100, library=LIB[::-1])
    assert _size(e1=e1)["verdict"] == "not_run"


def test_builds_judged_in_any_order():
    arm = _arm(bits=(60, 100, 80))
    arm["roots"].reverse()
    assert _size(arm)["bits"] == 140
