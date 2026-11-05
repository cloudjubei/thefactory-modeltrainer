"""Direct tests for harness/floor_x2.py — X2, the whole game's ply-6 builds rebuilt with every map also read on the
mirrored board: a mirrored reading plays a stored map, so it pays that map once with its plain reading and no more;
judged on the whole game with Z1's taken rebuilds as the control and the mirror rebuilds chosen as U1 chooses them."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_x2 as floor_x2
from harness.floor_x2 import SPEC, x2_library, x2_piece, x2_report, x2_size
from harness.floor_w6 import w6_report
from harness.floor_z1 import SPEC as Z1
from tests.test_floor_g4 import ASSEMBLY, G3, _row, _run
from tests.test_floor_z1 import BUILT, CONTROL as U1_CONTROL, FOUND, G4_LIBRARY, JUDGED as Z_JUDGED, LIBRARY
from tests.test_floor_z1 import _arm as _z_arm, _earlier as _z_earlier, _g4

READS = [*LIBRARY, *LIBRARY]
N = len(LIBRARY)
CONTROL = U1_CONTROL - 20
JUDGED = {**SPEC, "roots": 2, "indices": [0, 2], "root_moves": [[0, None], [2, None]], "kept": [12], "z_held": [0, 2],
          "control_bits": CONTROL}


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


@pytest.fixture(autouse=True)
def _costs(monkeypatch):
    costs = (40, 60, 11, 12, 13, 14, 15)
    monkeypatch.setattr(floor_x2, "map_bits", lambda levels, empty, level_bits: costs[
        [{int(c): k for c, k in m.items()} for m, _ in LIBRARY].index(levels)])


def _z(rows=None):
    return (_run(rows, Z_JUDGED) if rows else _z_arm()), Z_JUDGED


def _arm(bits=(40, 185), leaves=((), ()), incomplete=(), library=READS, root_moves=(None, None)):
    return _run([_row(i, b, leaves=l, library=library, complete=i not in incomplete, root_move=m)
                 for i, b, l, m in zip((0, 2), bits, leaves, root_moves)], JUDGED)


def _size(arm=None, z=None, earlier=None, g4=None, spec=JUDGED, assembly=ASSEMBLY):
    return x2_size(arm or _arm(), z or _z(), earlier or _z_earlier(), g4 or _g4(), G3, assembly, spec)


def test_the_spec_is_z1_s_rebuilds_with_the_mirror_on():
    assert SPEC["builder"] == {**Z1["builder"], "mirror": True}
    assert SPEC["root_moves"] == Z1["root_moves"] and SPEC["indices"] == Z1["indices"] and SPEC["roots"] == 51
    assert len(SPEC["z_held"]) == 27 and SPEC["z_held"] == sorted(SPEC["z_held"])
    assert set(SPEC["z_held"]) <= set(SPEC["indices"]) and SPEC["control_bits"] == 180_199
    assert SPEC["library_from"] == "c49_x2_library.json.gz" and x2_report is w6_report
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (48, 40)
    assert (SPEC["support"], SPEC["refute"]) == ((97, 100), (99, 100))


def test_the_library_comes_twice_mirrored_readings_first():
    assert x2_library([1, 2]) == [1, 2, 1, 2]


def test_both_readings_of_a_map_are_that_map_and_own_maps_are_none():
    r = _row(0, 50, leaves=(2, N + 2, N + 3, 2 * N + 1), library=READS, found=FOUND)
    assert x2_piece(r, LIBRARY) == (50, frozenset({2, 3}))


def test_a_smaller_rebuild_replaces_its_build():
    r = _size(arm=_arm(bits=(40, 190)))
    assert r == {"verdict": "inconclusive", "bits": CONTROL - 10, "control_bits": CONTROL,
                 "ratio": (CONTROL - 10) / CONTROL, "rebuilt": 1, "unfinished": False}


def test_a_larger_rebuild_leaves_its_build():
    r = _size(arm=_arm(bits=(60, 200)))
    assert r["verdict"] == "refuted" and r["bits"] == CONTROL and r["rebuilt"] == 0


def test_a_map_the_game_pays_read_either_way_costs_nothing():
    assert _size(arm=_arm(bits=(40, 190), leaves=((2,), ())))["bits"] == CONTROL - 10
    assert _size(arm=_arm(bits=(40, 190), leaves=((N + 2,), ())))["bits"] == CONTROL - 10


def test_a_new_map_read_both_ways_is_paid_once():
    index = N
    assert _size(arm=_arm(bits=(40, 180), leaves=((index,), (2 * index,))))["bits"] == CONTROL - 20 + 15


def test_only_the_rebuilds_z1_took_are_in_the_control():
    spec = {**JUDGED, "z_held": [0], "control_bits": U1_CONTROL - 10}
    r = _size(arm=_arm(bits=(60, 195)), spec=spec)
    assert r["control_bits"] == U1_CONTROL - 10 and r["bits"] == U1_CONTROL - 15 and r["rebuilt"] == 1


def test_the_smaller_of_the_two_starts_is_kept(monkeypatch):
    real = floor_x2.u1_choose

    def worse_from(start_empty):
        def choose(size, plays, start):
            held, bits = real(size, plays, start)
            return (frozenset(), 9_999) if (not start) == start_empty else (held, bits)

        return choose

    for start_empty in (True, False):
        monkeypatch.setattr(floor_x2, "u1_choose", worse_from(start_empty))
        r = _size()
        assert (r["bits"], r["rebuilt"]) == (CONTROL - 15, 2)


def test_an_unfinished_rebuild_keeps_its_build_and_cannot_refute():
    r = _size(arm=_arm(bits=(900, 900), incomplete=(0,)))
    assert r["verdict"] == "inconclusive" and r["bits"] == CONTROL and r["unfinished"] and r["rebuilt"] == 0
    assert _size(arm=_arm(bits=(900, 185), incomplete=(0,)))["bits"] == CONTROL - 5


def test_the_bars_are_inclusive():
    bits = CONTROL - 15
    assert _size(spec={**JUDGED, "support": (bits, CONTROL)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, CONTROL)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits, CONTROL)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits + 1, CONTROL)})["verdict"] == "inconclusive"


def test_a_run_off_its_registration_is_not_run():
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"})["verdict"] == "not_run"
    earlier = _z_earlier()
    earlier[0][0]["builder"] = {"other": 1}
    assert _size(earlier=earlier)["verdict"] == "not_run"
    run, spec = _g4()
    assert _size(g4=({**run, "builder": {"other": 1}}, spec))["verdict"] == "not_run"
    run, spec = _z()
    assert _size(z=({**run, "builder": {"other": 1}}, spec))["verdict"] == "not_run"
    assert _size(spec={**JUDGED, "control_bits": CONTROL + 1})["verdict"] == "not_run"


def test_builds_other_than_the_game_takes_are_not_run():
    spec = {**JUDGED, "roots": 1, "indices": [2], "root_moves": [[2, None]]}
    assert _size(arm=_run(_arm()["roots"][1:], spec), spec=spec)["verdict"] == "not_run"


def test_rebuilds_other_than_the_registered_are_not_run():
    arm = _arm()
    arm["roots"][1] = _row(1, 190, library=READS)
    assert _size(arm=arm)["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][0]["root_move"] = 3
    assert _size(arm=arm)["verdict"] == "not_run"


def test_a_rebuild_off_its_root_move_is_not_run():
    earlier = _z_earlier([_row(0, 60, found=BUILT[0], root_move=3), _row(2, 200, found=BUILT[2])])
    spec = {**JUDGED, "root_moves": [[0, 3], [2, None]]}
    g4 = _g4(spec={**_g4()[1], "taken": [[0, 0, 3], [0, 2, None]]})
    arm = _arm(root_moves=(3, None))
    assert _size(arm=arm, earlier=earlier, g4=g4, spec=spec)["bits"] == CONTROL - 15
    arm["roots"][0]["nodes"][0][2]["move"] = 4
    assert _size(arm=arm, earlier=earlier, g4=g4, spec=spec)["verdict"] == "not_run"


def test_a_row_at_another_position_than_its_index_is_not_run():
    arm = _arm()
    arm["roots"][1] = _row(2, 190, library=READS, at=1)
    assert _size(arm=arm)["verdict"] == "not_run"
    rows = [_row(0, 50, library=LIBRARY), _row(2, 190, library=LIBRARY, at=1)]
    assert _size(z=_z(rows))["verdict"] == "not_run"
    rows = [_row(0, 60, found=BUILT[0]), _row(2, 200, found=BUILT[2])]
    rows[1]["index"] = 1
    arm = _run([_row(0, 40, library=READS), _row(1, 185, library=READS)], JUDGED)
    z = _run([_row(0, 50, library=LIBRARY), _row(1, 190, library=LIBRARY)], Z_JUDGED)
    g4 = _g4(spec={**_g4()[1], "taken": [[0, 0, None], [0, 1, None]]})
    spec = {**JUDGED, "root_moves": [[0, None], [1, None]], "z_held": [0, 1]}
    assert _size(arm=arm, z=(z, Z_JUDGED), earlier=_z_earlier(rows), g4=g4, spec=spec)["verdict"] == "not_run"


def test_a_misnumbered_row_stops_the_judge_before_it_assembles(monkeypatch):
    def assembled(rows, assembly):
        raise AssertionError("assembled")

    monkeypatch.setattr(floor_x2, "v2_assemble", assembled)
    rows = [_row(0, 60, found=BUILT[0]), _row(2, 200, found=BUILT[2])]
    rows[1]["index"] = 1
    assert _size(earlier=_z_earlier(rows))["verdict"] == "not_run"
    assert _size(z=_z([_row(0, 50, library=LIBRARY), _row(1, 190, library=LIBRARY, at=2)]))["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][1]["index"] = 1
    assert _size(arm=arm)["verdict"] == "not_run"


def test_rows_off_their_library_are_not_run_unless_unfinished():
    assert _size(arm=_arm(library=LIBRARY))["verdict"] == "not_run"
    assert _size(arm=_arm(library=LIBRARY, incomplete=(0, 2)))["bits"] == CONTROL
    assert _size(z=_z([_row(0, 50, library=READS), _row(2, 190, library=LIBRARY)]))["verdict"] == "not_run"
    assert _size(earlier=_z_earlier([_row(0, 60, library=[], found=BUILT[0]),
                                     _row(2, 200, found=BUILT[2])]))["verdict"] == "not_run"
    rows = [_row(10, 90, library=G4_LIBRARY), _row(11, 200, library=G4_LIBRARY),
            _row(12, 45, leaves=(2,), library=[], found=FOUND)]
    assert _size(g4=_g4(rows))["verdict"] == "not_run"


def test_a_rebuild_z1_did_not_hold_or_finish_is_not_run():
    assert _size(spec={**JUDGED, "z_held": [0, 2, 3]})["verdict"] == "not_run"
    rows = [_row(0, 50, library=LIBRARY), _row(2, 190, library=LIBRARY, complete=False)]
    assert _size(z=_z(rows))["verdict"] == "not_run"
    assert _size(spec={**JUDGED, "kept": [12, 13]})["verdict"] == "not_run"
