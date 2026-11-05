"""Direct tests for harness/floor_z1.py — Z1, the assembled whole game's ply-6 builds rebuilt: the 51 builds it takes,
each rebuilt from every map it holds (G4's library, then the own maps of the rebuilds it keeps), judged on the whole
game with the rebuilds chosen as U1 chooses them."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_z1 as floor_z1
from harness.floor_g4 import SPEC as G4
from harness.floor_w6 import w6_report
from harness.floor_z1 import SPEC, z1_library, z1_report, z1_size
from tests.test_floor_g4 import ASSEMBLY, BUILT, G3, LIB, OWN, _row, _run

FOUND = [[{"2": 3}, 7]]
G4_LIBRARY = LIB + OWN[10] + OWN[11] + BUILT[0] + BUILT[2]
LIBRARY = G4_LIBRARY + FOUND
G4_JUDGED = {**G4, "roots": 3, "indices": [10, 11, 12], "taken": [[0, 0, None], [0, 2, None]]}
CONTROL = 4 + 4 + 300 + 45 + 1 + 60 + 60 + 200
JUDGED = {**SPEC, "roots": 2, "indices": [0, 2], "root_moves": [[0, None], [2, None]], "kept": [12],
          "control_bits": CONTROL}


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


@pytest.fixture(autouse=True)
def _costs(monkeypatch):
    costs = (40, 60, 11, 12, 13, 14, 15)
    monkeypatch.setattr(floor_z1, "map_bits", lambda levels, empty, level_bits: costs[
        [{int(c): k for c, k in m.items()} for m, _ in LIBRARY].index(levels)])


def _earlier(rows=None):
    rows = rows or [_row(0, 60, found=BUILT[0]), _row(2, 200, found=BUILT[2])]
    spec = {**JUDGED, "roots": len(rows)}
    return [(_run(rows, spec), spec)]


def _g4(rows=None, spec=G4_JUDGED):
    rows = rows or [_row(10, 90, library=G4_LIBRARY), _row(11, 200, library=G4_LIBRARY),
                    _row(12, 45, leaves=(2,), library=G4_LIBRARY, found=FOUND)]
    return _run(rows, spec), spec


def _arm(bits=(50, 190), leaves=((), ()), incomplete=(), library=LIBRARY, root_moves=(None, None)):
    return _run([_row(i, b, leaves=l, library=library, complete=i not in incomplete, root_move=m)
                 for i, b, l, m in zip((0, 2), bits, leaves, root_moves)], JUDGED)


def _size(arm=None, earlier=None, g4=None, spec=JUDGED, assembly=ASSEMBLY):
    return z1_size(arm or _arm(), earlier or _earlier(), g4 or _g4(), G3, assembly, spec)


def test_the_spec_names_the_builds_and_the_rebuilds_kept():
    assert SPEC["roots"] == len(SPEC["root_moves"]) == 51 == len({i for i, _ in SPEC["root_moves"]})
    assert SPEC["indices"] == sorted(i for i, _ in SPEC["root_moves"]) and len(SPEC["kept"]) == 66
    assert SPEC["kept"] == sorted(SPEC["kept"]) and set(SPEC["kept"]) <= set(G4["indices"])
    assert SPEC["builder"]["seconds"] == 21_600.0 and SPEC["frontier"] == 148
    assert SPEC["control_bits"] == 184_328 and SPEC["library_from"] == "c49_z1_library.json.gz"
    assert z1_report is w6_report


def test_the_library_is_g4_s_then_the_kept_rebuilds_own_maps():
    run, _ = _g4()
    assert z1_library(G4_LIBRARY, run, JUDGED) == LIBRARY
    assert z1_library(G4_LIBRARY, run, {**JUDGED, "kept": [10, 11]}) == G4_LIBRARY


def test_the_library_follows_the_positions_not_the_order_they_finished():
    rows = [_row(12, 45, found=FOUND), _row(10, 90, found=[[{"1": 1}, 7]])]
    assert z1_library([], _run(rows, G4_JUDGED), {**JUDGED, "kept": [10, 12]}) == [[{"1": 1}, 7], *FOUND]


def test_a_smaller_rebuild_replaces_its_build():
    r = _size(arm=_arm(bits=(50, 200)))
    assert r == {"verdict": "inconclusive", "bits": CONTROL - 10, "control_bits": CONTROL,
                 "ratio": (CONTROL - 10) / CONTROL, "rebuilt": 1, "unfinished": False}


def test_a_larger_rebuild_leaves_its_build():
    r = _size(arm=_arm(bits=(70, 210)))
    assert r["verdict"] == "refuted" and r["bits"] == CONTROL and r["rebuilt"] == 0


def test_a_map_the_game_already_pays_costs_a_rebuild_nothing():
    assert _size(arm=_arm(bits=(50, 200), leaves=((2,), ())))["bits"] == CONTROL - 10


def test_rebuilds_sharing_a_new_map_come_in_together_and_pay_it_once():
    assert _size(arm=_arm(leaves=((7,), (7,))))["bits"] == CONTROL - 20 + 15
    r = _size(arm=_arm(bits=(55, 195), leaves=((7,), (7,))))
    assert r["bits"] == CONTROL and r["rebuilt"] == 0


def test_a_kept_rebuild_s_own_piece_is_read_against_g4_s_library():
    rows = [_row(10, 90, library=G4_LIBRARY), _row(11, 200, library=G4_LIBRARY),
            _row(12, 45, leaves=(2, len(G4_LIBRARY) + 1), library=G4_LIBRARY, found=FOUND)]
    assert _size(g4=_g4(rows))["bits"] == CONTROL - 20


def test_the_smaller_of_the_two_starts_is_kept(monkeypatch):
    real = floor_z1.u1_choose

    def worse_from(start_empty):
        def choose(size, plays, start):
            held, bits = real(size, plays, start)
            return (frozenset(), 9_999) if (not start) == start_empty else (held, bits)

        return choose

    for start_empty in (True, False):
        monkeypatch.setattr(floor_z1, "u1_choose", worse_from(start_empty))
        r = _size()
        assert (r["bits"], r["rebuilt"]) == (CONTROL - 20, 2)


def test_an_unfinished_rebuild_keeps_its_build_and_cannot_refute():
    r = _size(arm=_arm(bits=(900, 900), incomplete=(0,)))
    assert r["verdict"] == "inconclusive" and r["bits"] == CONTROL and r["unfinished"] and r["rebuilt"] == 0
    assert _size(arm=_arm(bits=(900, 190), incomplete=(0,)))["bits"] == CONTROL - 10


def test_the_bars_are_inclusive():
    bits = CONTROL - 20
    assert _size(spec={**JUDGED, "support": (bits, CONTROL)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, CONTROL)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits, CONTROL)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits + 1, CONTROL)})["verdict"] == "inconclusive"


def test_a_run_off_its_registration_is_not_run():
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"})["verdict"] == "not_run"
    earlier = _earlier()
    earlier[0][0]["builder"] = {"other": 1}
    assert _size(earlier=earlier)["verdict"] == "not_run"
    run, spec = _g4()
    assert _size(g4=({**run, "builder": {"other": 1}}, spec))["verdict"] == "not_run"
    assert _size(spec={**JUDGED, "control_bits": CONTROL + 1})["verdict"] == "not_run"


def test_builds_other_than_the_game_takes_are_not_run():
    spec = {**JUDGED, "roots": 1, "indices": [2], "root_moves": [[2, None]]}
    assert _size(arm=_run(_arm()["roots"][1:], spec), spec=spec)["verdict"] == "not_run"


def test_rebuilds_other_than_the_registered_are_not_run():
    arm = _arm()
    arm["roots"][1] = _row(1, 190, library=LIBRARY)
    assert _size(arm=arm)["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][0]["root_move"] = 3
    assert _size(arm=arm)["verdict"] == "not_run"


def test_a_rebuild_off_its_root_move_is_not_run():
    earlier = _earlier([_row(0, 60, found=BUILT[0], root_move=3), _row(2, 200, found=BUILT[2])])
    spec = {**JUDGED, "root_moves": [[0, 3], [2, None]]}
    g4 = _g4(spec={**G4_JUDGED, "taken": [[0, 0, 3], [0, 2, None]]})
    assert _size(arm=_arm(root_moves=(3, None)), earlier=earlier, g4=g4, spec=spec)["bits"] == CONTROL - 20
    arm = _arm(root_moves=(3, None))
    arm["roots"][0]["nodes"][0][2]["move"] = 4
    assert _size(arm=arm, earlier=earlier, g4=g4, spec=spec)["verdict"] == "not_run"


def test_a_row_at_another_position_than_its_index_is_not_run():
    arm = _arm()
    arm["roots"][1] = _row(2, 190, library=LIBRARY, at=1)
    assert _size(arm=arm)["verdict"] == "not_run"
    rows = [_row(0, 60, found=BUILT[0]), _row(2, 200, found=BUILT[2])]
    rows[1]["index"] = 1
    arm = _run([_row(0, 50, library=LIBRARY), _row(1, 190, library=LIBRARY)], JUDGED)
    g4 = _g4(spec={**G4_JUDGED, "taken": [[0, 0, None], [0, 1, None]]})
    spec = {**JUDGED, "root_moves": [[0, None], [1, None]]}
    assert _size(arm=arm, earlier=_earlier(rows), g4=g4, spec=spec)["verdict"] == "not_run"


def test_rows_off_their_library_are_not_run_unless_unfinished():
    assert _size(arm=_arm(library=G4_LIBRARY))["verdict"] == "not_run"
    assert _size(arm=_arm(library=G4_LIBRARY, incomplete=(0, 2)))["bits"] == CONTROL
    assert _size(earlier=_earlier([_row(0, 60, library=LIB[:1], found=BUILT[0]),
                                   _row(2, 200, found=BUILT[2])]))["verdict"] == "not_run"
    rows = [_row(10, 90, library=G4_LIBRARY), _row(11, 200, library=G4_LIBRARY),
            _row(12, 45, leaves=(2,), library=LIB, found=FOUND)]
    assert _size(g4=_g4(rows))["verdict"] == "not_run"


def test_a_kept_rebuild_g4_does_not_hold_or_finish_is_not_run():
    assert _size(spec={**JUDGED, "kept": [12, 13]})["verdict"] == "not_run"
    rows = [_row(10, 90, library=G4_LIBRARY), _row(11, 200, library=G4_LIBRARY),
            _row(12, 45, leaves=(2,), library=G4_LIBRARY, found=FOUND, complete=False)]
    assert _size(g4=_g4(rows))["verdict"] == "not_run"
