"""Direct tests for harness/floor_w3.py — W3, W2's two unfinished frontier positions rebuilt for up to 6 hours from
W2's library, deciding the whole-game projection against the net. Synthetic evidence."""
from __future__ import annotations

import copy

from harness.floor_w2 import SPEC as W2
from harness.floor_w3 import SPEC, w3_report

W2_ROOTS = [{"index": i, "board": [i], "complete": done, "bits": {"nodes": bits}, "found_maps": [[{}, 0]] * found}
            for i, done, bits, found in ((362, True, 1_000, 1), (87, True, 1_000, 1), (316, False, 900, 3),
                                         (346, True, 1_000, 1), (230, False, 600, 2), (323, True, 1_000, 2),
                                         (393, True, 1_000, 0), (263, True, 1_000, 0))]
LIBRARY = 10


def _root(index, complete, nodes, shared=LIBRARY):
    return {"index": index, "board": [index], "complete": complete, "bits": {"nodes": nodes}, "shared": shared,
            "checked": True if complete else None, "moves_win": True if complete else None,
            "leaves_certified": True if complete else None}


def _evidence(roots):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": copy.deepcopy(SPEC["builder"]), "opening": {"moves": 204, "trivial": 259, "frontier": 671},
            "roots": roots}


def _w2():
    return {"roots": copy.deepcopy(W2_ROOTS)}


def test_w3_rebuilds_w2_s_two_unfinished_roots_for_6_hours_from_w2_s_library():
    assert SPEC["indices"] == [316, 230] and SPEC["library_from"] == "c49_w2.json.gz"
    assert SPEC["builder"] == {**W2["builder"], "seconds": 21600.0} and SPEC["projection_line"] == 660_000
    assert "wave_size" not in SPEC


def test_both_finished_and_small_decides_the_projection_under_the_net():
    r = w3_report(_evidence([_root(316, True, 1_000), _root(230, True, 1_000)]), SPEC, _w2())
    assert r["integrity"] == [] and r["projection"]["bits"] == 204 * 4 + 259 + 671 * 1_000
    assert r["projection"]["verdict"] == "refuted" and r["projection"]["lower_bound"] is False
    r = w3_report(_evidence([_root(316, True, 900), _root(230, True, 900)]), SPEC, _w2())
    assert r["projection"]["verdict"] == "supported" and r["complete"] == {"verdict": "supported", "count": 2}


def test_an_unfinished_root_keeps_a_small_projection_undecided_but_a_large_one_refutes():
    small = w3_report(_evidence([_root(316, False, 900), _root(230, True, 900)]), SPEC, _w2())
    assert small["projection"]["verdict"] == "inconclusive" and small["projection"]["lower_bound"] is True
    assert small["complete"]["verdict"] == "inconclusive"
    large = w3_report(_evidence([_root(316, False, 9_000), _root(230, False, 900)]), SPEC, _w2())
    assert large["projection"]["verdict"] == "refuted" and large["complete"]["verdict"] == "refuted"


def test_the_redone_roots_replace_w2_s_and_the_rest_are_w2_s():
    r = w3_report(_evidence([_root(316, True, 100), _root(230, True, 300)]), SPEC, _w2())
    assert r["projection"]["bits"] == 204 * 4 + 259 + 671 * (6 * 1_000 + 100 + 300) / 8


def test_integrity_problems_stop_the_judgement():
    good = [_root(316, True, 900), _root(230, True, 900)]
    moved = copy.deepcopy(good)
    moved[0]["board"] = [1]
    r = w3_report(_evidence(moved), SPEC, _w2())
    assert r["integrity"] and r["projection"] == {"verdict": "not_run"}
    no_library = copy.deepcopy(good)
    no_library[1]["shared"] = 0
    assert w3_report(_evidence(no_library), SPEC, _w2())["integrity"]
    other = _evidence(copy.deepcopy(good))
    other["builder"]["seconds"] = 7200.0
    assert w3_report(other, SPEC, _w2())["integrity"]
    wrong = _evidence([_root(316, True, 900), _root(87, True, 900)])
    assert w3_report(wrong, SPEC, _w2())["integrity"]
    unchecked = copy.deepcopy(good)
    unchecked[0]["leaves_certified"] = False
    assert w3_report(_evidence(unchecked), SPEC, _w2())["integrity"]
    foreign = _evidence(copy.deepcopy(good))
    foreign["measurement_fingerprint"] = "x"
    assert w3_report(foreign, SPEC, _w2())["integrity"]
