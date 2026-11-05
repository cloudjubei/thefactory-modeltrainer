"""Direct tests for harness/floor_x4.py — X4, the control X3 did not have: the rebuilds that played a mirrored leaf,
rebuilt from the same list of maps with the mirror off, and the whole game judged with them in X3's place."""
from __future__ import annotations

import pytest

import harness.floor_j1 as floor_j1
import harness.floor_x2 as floor_x2
import harness.floor_x3 as floor_x3
import harness.floor_x4 as floor_x4
from harness.floor_g4 import SPEC as G4
from harness.floor_w6 import w6_report
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC, x4_report, x4_size
from tests.test_floor_g4 import ASSEMBLY, G3, _row, _run
from tests.test_floor_x2 import LIBRARY, N, READS
from tests.test_floor_x3 import CONTROL as X2_GAME, JUDGED as X3_JUDGED, _arm as _x3_arm, _x2, _z
from tests.test_floor_z1 import _earlier, _g4

GAME = X2_GAME - 15
JUDGED = {**SPEC, "roots": 1, "indices": [12], "control_bits": GAME, "support": (GAME + 5, GAME),
          "refute": (GAME + 1, GAME)}


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


def _arm(bits=60, leaves=(N + 2,), complete=True, library=READS, index=12, at=None):
    return _run([_row(index, bits, leaves=leaves, library=library, complete=complete, at=at)], JUDGED)


def _size(arm=None, x3=None, spec=JUDGED):
    return x4_size(arm or _arm(), x3 or (_x3_arm(), X3_JUDGED), _x2(), _z(), _earlier(), _g4(), G3, ASSEMBLY, spec)


def test_the_spec_is_x3_s_mirrored_rebuilds_with_the_mirror_off():
    assert SPEC["builder"] == G4["builder"] and "mirror" not in SPEC["builder"]
    assert SPEC["roots"] == len(SPEC["indices"]) == 18 and SPEC["indices"] == sorted(SPEC["indices"])
    assert set(SPEC["indices"]) <= set(X3["indices"]) and SPEC["library_from"] == X3["library_from"]
    assert SPEC["measurement_fp"] == X3["measurement_fp"] and SPEC["control_bits"] == 166_623
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (18, 16) and x4_report is w6_report
    assert (SPEC["support"], SPEC["refute"]) == ((101, 100), (401, 400))


def test_the_game_with_the_mirror_off_stands_in_for_x3_s_rebuilds():
    assert _size() == {"verdict": "supported", "bits": GAME + 5, "control_bits": GAME, "ratio": (GAME + 5) / GAME,
                       "rebuilt": 1, "unfinished": False}


def test_a_control_rebuild_as_small_as_x3_s_gives_the_mirror_nothing():
    r = _size(arm=_arm(bits=40))
    assert r["verdict"] == "refuted" and r["bits"] == GAME and r["rebuilt"] == 2


def test_a_control_rebuild_that_plays_a_mirrored_leaf_is_not_run():
    assert _size(arm=_arm(leaves=(2,)))["verdict"] == "not_run"
    assert _size(arm=_arm(leaves=(N,)))["verdict"] == "not_run"
    assert _size(arm=_arm(leaves=(1,)))["verdict"] == "not_run"
    assert _size(arm=_arm(leaves=(N + 1,)))["verdict"] != "not_run"
    assert _size(arm=_arm(leaves=(0,)))["verdict"] != "not_run"


def test_a_mirrored_leaf_in_an_unfinished_rebuild_does_not_count():
    assert _size(arm=_arm(leaves=(2,), complete=False))["verdict"] != "not_run"


def test_an_unfinished_rebuild_cannot_support_but_can_refute():
    r = _size(arm=_arm(complete=False))
    assert r["verdict"] == "inconclusive" and r["bits"] == GAME + 5 and r["unfinished"]
    spec = {**JUDGED, "refute": (GAME + 5, GAME)}
    assert _size(arm=_arm(complete=False), spec=spec)["verdict"] == "refuted"


def test_the_bars_are_inclusive():
    bits = GAME + 5
    assert _size(spec={**JUDGED, "support": (bits + 1, GAME)})["verdict"] == "inconclusive"
    assert _size(spec={**JUDGED, "refute": (bits, GAME)})["verdict"] == "refuted"
    assert _size(spec={**JUDGED, "refute": (bits - 1, GAME)})["verdict"] == "supported"


def test_a_run_off_its_registration_is_not_run():
    assert _size(arm={**_arm(), "measurement_fingerprint": "other"})["verdict"] == "not_run"
    assert _size(arm={**_arm(), "builder": X3["builder"]})["verdict"] == "not_run"


def test_rows_other_than_the_registered_are_not_run():
    assert _size(arm=_arm(index=11))["verdict"] == "not_run"
    assert _size(arm=_arm(at=13))["verdict"] == "not_run"


def test_rows_off_x3_s_library_are_not_run():
    assert _size(arm=_arm(library=LIBRARY, leaves=()))["verdict"] == "not_run"


def test_a_game_other_than_x3_s_is_not_run(monkeypatch):
    assert _size(spec={**JUDGED, "control_bits": GAME + 1})["verdict"] == "not_run"
    assert _size(x3=(_x3_arm(bits=(290, 45)), X3_JUDGED))["verdict"] == "not_run"
    real = floor_x4.x3_size
    monkeypatch.setattr(floor_x4, "x3_size", lambda e, *a: {"verdict": "not_run"}
                        if e["roots"][1]["bits"]["nodes"] == 60 else real(e, *a))
    assert _size()["verdict"] == "not_run"
