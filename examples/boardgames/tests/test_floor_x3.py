"""Direct tests for harness/floor_x3.py — X3, the whole game's ply-8 strategies rebuilt with every map also read on the
mirrored board: the positions G4 rebuilt, judged on the whole game with X2's game as the control and the rebuilds
chosen as U1 chooses them."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_x2 as floor_x2
import harness.floor_x3 as floor_x3
from harness.floor_g4 import SPEC as G4
from harness.floor_w6 import w6_report
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC, x3_report, x3_size
from tests.test_floor_g4 import ASSEMBLY, G3, _row, _run
from tests.test_floor_x2 import CONTROL as Z_CONTROL, JUDGED as X2_JUDGED, LIBRARY, N, READS, _arm as _x2_arm, _z
from tests.test_floor_z1 import _earlier, _g4

CONTROL = Z_CONTROL - 15
JUDGED = {**SPEC, "roots": 2, "indices": [11, 12], "kept": [12], "z_held": [0, 2], "x2_held": [0, 2],
          "control_bits": CONTROL}


@pytest.fixture(autouse=True)
def _bits_as_coded(monkeypatch):
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"])


@pytest.fixture(autouse=True)
def _costs(monkeypatch):
    costs = (40, 60, 11, 12, 13, 14, 15)

    def cost(levels, empty, level_bits):
        return costs[[{int(c): k for c, k in m.items()} for m, _ in LIBRARY].index(levels)]

    monkeypatch.setattr(floor_x2, "map_bits", cost)
    monkeypatch.setattr(floor_x3, "map_bits", cost)


def _arm(bits=(290, 40), leaves=((), (2,)), incomplete=(), library=READS):
    return _run([_row(i, b, leaves=l, library=library, complete=i not in incomplete)
                 for i, b, l in zip((11, 12), bits, leaves)], JUDGED)


def _x2(arm=None):
    return arm or _x2_arm(), X2_JUDGED


def _size(arm=None, x2=None, spec=JUDGED, assembly=ASSEMBLY):
    return x3_size(arm or _arm(), x2 or _x2(), _z(), _earlier(), _g4(), G3, assembly, spec)


def test_the_spec_is_g4_s_rebuilds_with_the_mirror_on():
    assert SPEC["builder"] == {**G4["builder"], "mirror": True}
    assert SPEC["indices"] == G4["indices"] and SPEC["roots"] == 409 and SPEC["frontier"] == G4["frontier"]
    assert SPEC["measurement_fp"] == X2["measurement_fp"] and SPEC["library_from"] == X2["library_from"]
    assert SPEC["kept"] == X2["kept"] and SPEC["z_held"] == X2["z_held"]
    assert SPEC["x2_held"] == [2, 4, 68, 71, 72, 118, 130] and SPEC["control_bits"] == 175_425
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (409, 401) and x3_report is w6_report
    assert (SPEC["support"], SPEC["refute"]) == ((97, 100), (99, 100))


def test_a_smaller_rebuild_replaces_its_strategy():
    r = _size(arm=_arm(bits=(290, 45)))
    assert r == {"verdict": "inconclusive", "bits": CONTROL - 10, "control_bits": CONTROL,
                 "ratio": (CONTROL - 10) / CONTROL, "rebuilt": 1, "unfinished": False}


def test_a_larger_rebuild_leaves_its_strategy():
    r = _size(arm=_arm(bits=(310, 50)))
    assert r["verdict"] == "refuted" and r["bits"] == CONTROL and r["rebuilt"] == 0


def test_a_map_the_game_pays_read_either_way_costs_nothing():
    assert _size(arm=_arm(bits=(290, 45), leaves=((2,), (2,))))["bits"] == CONTROL - 10
    assert _size(arm=_arm(bits=(290, 45), leaves=((N + 2,), (2,))))["bits"] == CONTROL - 10
    assert _size(arm=_arm(bits=(290, 45), leaves=((), (N + 2,))))["bits"] == CONTROL - 10


def test_a_new_map_read_both_ways_is_paid_once():
    assert _size(arm=_arm(bits=(280, 30), leaves=((N,), (2 * N, 2))))["bits"] == CONTROL - 35 + 15


def test_only_the_rebuilds_x2_took_are_in_the_control():
    arm = _x2_arm(bits=(40, 190))
    spec = {**JUDGED, "x2_held": [0], "control_bits": Z_CONTROL - 10}
    r = _size(arm=_arm(bits=(310, 50)), x2=_x2(arm), spec=spec)
    assert r["control_bits"] == Z_CONTROL - 10 and r["bits"] == Z_CONTROL - 10


def test_the_rebuilds_x2_took_replace_z1_s():
    r = _size(arm=_arm(bits=(310, 50)), spec={**JUDGED, "x2_held": [0, 2]})
    assert r["control_bits"] == CONTROL
    spec = {**JUDGED, "x2_held": [2], "control_bits": Z_CONTROL - 5}
    assert _size(arm=_arm(bits=(310, 50)), spec=spec)["verdict"] == "not_run"


def test_only_the_rebuilds_z1_took_are_in_the_control():
    x2_spec = {**X2_JUDGED, "z_held": [0], "control_bits": Z_CONTROL + 10}
    spec = {**JUDGED, "z_held": [0], "x2_held": [], "control_bits": Z_CONTROL + 10}
    r = _size(arm=_arm(bits=(310, 50)), x2=(_x2_arm(bits=(60, 200)), x2_spec), spec=spec)
    assert r["control_bits"] == r["bits"] == Z_CONTROL + 10


def test_a_build_x2_took_is_priced_by_the_map_it_plays_whichever_way_it_reads():
    arm = _x2_arm(leaves=((N + 2,), ()))
    assert _size(arm=_arm(bits=(310, 50)), x2=_x2(arm))["bits"] == CONTROL


def test_the_smaller_of_the_two_starts_is_kept(monkeypatch):
    real = floor_x3.u1_choose

    def worse_from(start_empty):
        def choose(size, plays, start):
            held, bits = real(size, plays, start)
            return (frozenset(), 9_999) if (not start) == start_empty else (held, bits)

        return choose

    for start_empty in (True, False):
        monkeypatch.setattr(floor_x3, "u1_choose", worse_from(start_empty))
        r = _size()
        assert (r["bits"], r["rebuilt"]) == (CONTROL - 15, 2)


def test_an_unfinished_rebuild_keeps_its_strategy_and_cannot_refute():
    r = _size(arm=_arm(bits=(900, 900), incomplete=(11,)))
    assert r["verdict"] == "inconclusive" and r["bits"] == CONTROL and r["unfinished"] and r["rebuilt"] == 0
    assert _size(arm=_arm(bits=(900, 40), incomplete=(11,)))["bits"] == CONTROL - 5


def test_the_bars_are_inclusive():
    bits = CONTROL - 15
    assert _size(spec={**JUDGED, "support": (bits, CONTROL)})["verdict"] == "supported"
    assert _size(spec={**JUDGED, "support": (bits - 1, CONTROL)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits, CONTROL)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "support": (1, 9), "refute": (bits + 1, CONTROL)})["verdict"] == "inconclusive"


def test_a_run_off_its_registration_is_not_run():
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"})["verdict"] == "not_run"
    assert _size(x2=({**_x2_arm(), "builder": {"other": 1}}, X2_JUDGED))["verdict"] == "not_run"


def test_a_game_other_than_x2_s_is_not_run(monkeypatch):
    assert _size(x2=_x2(_x2_arm(bits=(40, 190))))["verdict"] == "not_run"
    monkeypatch.setattr(floor_x3, "x2_size", lambda *args: {"verdict": "inconclusive", "bits": CONTROL + 1})
    assert _size()["verdict"] == "not_run"


def test_rebuilds_other_than_the_registered_are_not_run():
    arm = _arm()
    arm["roots"][1] = _row(13, 40, library=READS)
    assert _size(arm=arm)["verdict"] == "not_run"


def test_a_row_at_another_position_than_its_index_is_not_run():
    arm = _arm()
    arm["roots"][1] = _row(12, 40, library=READS, at=13)
    assert _size(arm=arm)["verdict"] == "not_run"


def test_rows_off_their_library_are_not_run_unless_unfinished():
    assert _size(arm=_arm(library=LIBRARY))["verdict"] == "not_run"
    assert _size(arm=_arm(library=LIBRARY, incomplete=(11, 12)))["bits"] == CONTROL


def test_a_rebuild_x2_did_not_hold_or_finish_is_not_run():
    assert _size(spec={**JUDGED, "x2_held": [0, 2, 3]})["verdict"] == "not_run"
    arm = _x2_arm(bits=(40, 190), incomplete=(2,))
    assert _size(x2=_x2(arm), spec={**JUDGED, "control_bits": Z_CONTROL - 10})["verdict"] == "not_run"
    assert _size(x2=_x2(arm), spec={**JUDGED, "x2_held": [0], "control_bits": Z_CONTROL - 10})["bits"] == Z_CONTROL - 25


def test_a_control_other_than_the_spec_s_is_not_run(monkeypatch):
    monkeypatch.setattr(floor_x3, "x2_size", lambda *args: {"verdict": "inconclusive", "bits": CONTROL + 1})
    assert _size(spec={**JUDGED, "control_bits": CONTROL + 1})["verdict"] == "not_run"
