"""Direct tests for harness/floor_v2.py — V2, every winning move by transposition group: the groups of ply-6 positions
on which the assembled whole game still pays most, every winning move not yet built at each member and a free build
at each single-move member never built, and the whole game re-assembled by J1's chooser and then by group (J2)."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_v2 as floor_v2
from harness.floor_v2 import SPEC, v2_assemble, v2_report, v2_selected, v2_size
from harness.floor_w6 import w6_report

LIB = [({"3": 0}, 42), ({"5": 1}, 42)]
WINNING = {0: [1, 3], 1: [2], 2: [4, 5], 3: [0, 6], 4: [0]}
MOVES = [[0, 3], [1, None], [2, 4], [2, 5], [4, None]]


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


def _key(index):
    return (index,) + (0,) * 41, 0


def _row(index, move, bits, complete=True, root_move=None, library=LIB, leaf_root=False, at=None):
    board = list(_key(index if at is None else at)[0])
    root = {"leaf": 1} if leaf_root else {"move": move}
    return {"index": index, "root_move": root_move, "complete": complete, "bits": {"nodes": bits},
            "board": board, "to_move": 0, "nodes": [[board, 0, root], [[index, 1] + [0] * 40, 0, {"leaf": 0}]],
            "maps": [{}, *[m for m, _ in library]], "map_empty": [0, *[e for _, e in library]],
            "shared": len(library), "checked": True if complete else None,
            "moves_win": True if complete else None, "leaves_certified": True if complete else None}


def _run(roots, spec):
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"frontier": 148}, "roots": roots}


def _earlier(rows=None):
    rows = rows or [_row(0, 1, 4000), _row(3, 0, 50)]
    spec = {**SPEC, "roots": len(rows)}
    return [(_run(rows, spec), spec)]


def _scope(**over):
    return {"positions_fp": SPEC["positions_fp"], "library_from": SPEC["library_from"],
            "positions": [{"index": i, "winning": w} for i, w in WINNING.items()], **over}


ASSEMBLY = {"opening": {"moves": 1, "trivial": 0},
            "below": {_key(0): ["s", "a0"], _key(1): ["s", "a1"], _key(2): ["a2", "t"], _key(3): ["a3"],
                      _key(4): ["s", "a4"]},
            "base": {"s": (3000, frozenset()), **{f"a{i}": (10, frozenset()) for i in (0, 1, 4)},
                     "a2": (5000, frozenset()), "a3": (2000, frozenset())},
            "map_cost": {1: 100, 2: 200}, "library": LIB}
CONTROL = 4 + 4 * 4 + 3000 + 3 * 10 + 5000 + 1 + 50
JUDGED = {**SPEC, "roots": 5, "indices": [0, 1, 2, 4], "root_moves": MOVES, "min_group_bits": 1000,
          "control_bits": CONTROL}


def _arm(bits=(30, 40, 6000, 4000, 20), incomplete=(), moves=MOVES):
    return _run([_row(i, 9 if m is None else m, b, complete=k not in incomplete, root_move=m)
                 for k, ((i, m), b) in enumerate(zip(moves, bits))], JUDGED)


def _size(arm=None, earlier=None, spec=JUDGED, scope=None, assembly=ASSEMBLY):
    return v2_size(arm or _arm(), earlier or _earlier(), scope or _scope(), assembly, spec)


def test_the_spec_names_the_moves_its_selection_finds():
    assert SPEC["roots"] == len(SPEC["root_moves"])
    assert SPEC["indices"] == sorted({i for i, _ in SPEC["root_moves"]})
    assert v2_report is w6_report


def test_the_assembly_is_j1_s_choice_then_by_group(monkeypatch):
    rows = [r for e, _ in _earlier() for r in e["roots"]]
    assert v2_assemble(rows, ASSEMBLY)[1] == CONTROL
    twins = [_row(2, 4, 100, root_move=4), _row(2, 5, 100, root_move=5)]
    assert v2_assemble(twins, ASSEMBLY)[0] == {_key(2): 0}
    monkeypatch.setattr(floor_v2, "j1_choose", lambda *args: ({_key(2): 1}, 0))
    assert v2_assemble(twins, ASSEMBLY)[0] == {_key(2): 1}


def test_the_selection_is_every_unbuilt_move_in_the_groups_the_game_still_pays_for():
    rows = [r for e, _ in _earlier() for r in e["roots"]]
    assert v2_selected(_scope(), rows, ASSEMBLY, JUDGED) == MOVES


def test_a_group_is_weighed_by_what_the_game_pays_there_now():
    rows = [r for e, _ in _earlier() for r in e["roots"]]
    assert v2_selected(_scope(), rows, ASSEMBLY, {**JUDGED, "min_group_bits": 3042}) == MOVES
    assert v2_selected(_scope(), rows, ASSEMBLY, {**JUDGED, "min_group_bits": 3043}) == [[2, 4], [2, 5]]
    assert v2_selected(_scope(), rows, ASSEMBLY, {**JUDGED, "min_group_bits": 5005}) == [[2, 4], [2, 5]]
    assert v2_selected(_scope(), rows, ASSEMBLY, {**JUDGED, "min_group_bits": 5006}) == []
    assert v2_selected(_scope(), rows, ASSEMBLY, {**JUDGED, "min_group_bits": 51}) == MOVES
    assert v2_selected(_scope(), rows, ASSEMBLY, {**JUDGED, "min_group_bits": 50}) == MOVES[:4] + [[3, 6]] + MOVES[4:]


def test_a_single_move_position_already_built_is_not_built_again():
    rows = [r for e, _ in _earlier() for r in e["roots"]] + [_row(1, 2, 5000, leaf_root=True)]
    assert v2_selected(_scope(), rows, ASSEMBLY, JUDGED) == [[0, 3], [2, 4], [2, 5], [4, None]]


def test_the_selection_needs_the_registered_positions_and_library():
    assert v2_selected(_scope(positions_fp="other"), [], ASSEMBLY, JUDGED) is None
    assert v2_selected(_scope(library_from="other"), [], ASSEMBLY, JUDGED) is None


def test_the_whole_game_swaps_a_group_together_and_is_judged_against_the_earlier_assembly():
    r = _size()
    bits = 4 + 30 + 40 + 20 + 4000 + 50
    assert r == {"verdict": "supported", "bits": bits, "control_bits": CONTROL, "ratio": bits / CONTROL,
                 "taken": [[0, 3], [1, None], [2, 5], [4, None]], "unfinished": False}


def test_no_saving_once_every_build_finished_refutes():
    r = _size(arm=_arm(bits=(5000, 5000, 6000, 6000, 5000)))
    assert r["verdict"] == "refuted" and r["bits"] == CONTROL and r["taken"] == []


def test_an_unfinished_build_is_never_taken_and_cannot_refute():
    r = _size(arm=_arm(bits=(5000, 5000, 6000, 1, 5000), incomplete=(3,)))
    assert r["verdict"] == "inconclusive" and r["bits"] == CONTROL and r["unfinished"]


def test_the_bars_are_inclusive():
    bits = 4 + 30 + 40 + 20 + 4000 + 50
    assert _size(spec={**JUDGED, "support": (bits, CONTROL)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, CONTROL)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits, CONTROL)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits + 1, CONTROL)})["verdict"] == "inconclusive"


def test_the_game_is_never_worse_than_the_earlier_assembly(monkeypatch):
    real = floor_v2.v2_assemble

    def worse(rows, assembly):
        chosen, bits = real(rows, assembly)
        return (chosen, bits) if len(rows) == 2 else ({_key(2): len(rows) - 2}, CONTROL + 9)

    monkeypatch.setattr(floor_v2, "v2_assemble", worse)
    r = _size()
    assert r["bits"] == CONTROL and r["taken"] == [] and r["verdict"] == "refuted"


def test_a_tie_keeps_the_earlier_build():
    earlier = _earlier([_row(0, 1, 4000), _row(3, 0, 50), _row(2, 4, 4000, root_move=4)])
    spec = {**JUDGED, "roots": 4, "root_moves": [m for m in MOVES if m != [2, 4]], "control_bits": CONTROL - 1005}
    arm = _arm(bits=(5000, 5000, 4000, 5000), moves=spec["root_moves"])
    r = _size(arm=arm, earlier=earlier, spec=spec)
    assert r["taken"] == [] and r["bits"] == CONTROL - 1005


def test_a_run_off_its_registration_is_not_run():
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"})["verdict"] == "not_run"
    earlier = _earlier()
    earlier[0][0]["builder"] = {"other": 1}
    assert _size(earlier=earlier)["verdict"] == "not_run"


def test_moves_other_than_the_selection_s_are_not_run():
    assert _size(spec={**JUDGED, "root_moves": MOVES[:4]})["verdict"] == "not_run"
    four = {**JUDGED, "roots": 4, "root_moves": MOVES[:4]}
    assert _size(arm=_arm(moves=MOVES[:4]), spec=four)["verdict"] == "not_run"
    assert _size(arm=_arm(moves=[[0, 3], [1, None], [2, 4], [2, 4], [4, None]]))["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][2]["nodes"][0][2]["move"] = 5
    assert _size(arm=arm)["verdict"] == "not_run"


def test_a_free_build_may_play_any_root():
    arm = _arm()
    arm["roots"][1] = _row(1, 2, 40, root_move=None, leaf_root=True)
    assert _size(arm=arm)["verdict"] == "supported"


def test_a_row_at_another_position_than_its_index_is_not_run():
    arm = _arm()
    arm["roots"][2] = _row(2, 4, 6000, root_move=4, at=3)
    assert _size(arm=arm)["verdict"] == "not_run"
    rows = [_row(0, 1, 4000), _row(3, 0, 50)]
    rows[1]["index"] = 2
    assert _size(earlier=_earlier(rows))["verdict"] == "not_run"


def test_pieces_off_the_library_are_not_run_unless_unfinished():
    other = [LIB[1], LIB[0]]
    arm = _arm()
    arm["roots"][3] = _row(2, 5, 4000, root_move=5, library=other)
    assert _size(arm=arm)["verdict"] == "not_run"
    arm["roots"][3] = _row(2, 5, 4000, root_move=5, library=other, complete=False)
    assert _size(arm=arm)["taken"] == [[0, 3], [1, None], [4, None]]
    rows = [_row(0, 1, 4000, library=other), _row(3, 0, 50)]
    assert _size(earlier=_earlier(rows))["verdict"] == "not_run"


def test_an_earlier_assembly_other_than_registered_is_not_run():
    assert _size(spec={**JUDGED, "control_bits": CONTROL + 1})["verdict"] == "not_run"
