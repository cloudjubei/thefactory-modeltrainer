"""Direct tests for harness/floor_w4.py — W4, W3 with the leaves walked in C, judged against W3 on time to finish
and on finishing what W3 left unfinished. Synthetic evidence."""
from __future__ import annotations

import copy

from harness.floor_w3 import SPEC as W3
from harness.floor_w4 import SPEC, w4_report


def _root(index, complete, seconds, nodes=1_000, shared=24):
    return {"index": index, "board": [index], "complete": complete, "build_seconds": seconds,
            "bits": {"nodes": nodes}, "shared": shared, "checked": True if complete else None,
            "moves_win": True if complete else None, "leaves_certified": True if complete else None}


W3_ROOTS = [_root(316, True, 15_000.0, 9_057), _root(230, False, 21_600.0, 5_446)]


def _evidence(roots):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "builder": copy.deepcopy(SPEC["builder"]),
            "search": copy.deepcopy(SPEC["search"]), "roots": roots}


def _w3():
    return {"roots": copy.deepcopy(W3_ROOTS)}


def test_w4_changes_only_the_walk_from_w3():
    assert {k: v for k, v in SPEC.items() if k not in ("measurement_fp", "search", "time_support")} == \
        {k: v for k, v in W3.items() if k not in ("measurement_fp", "search")}
    assert SPEC["search"] == {**W3["search"], "walker": "native"} and SPEC["time_support"] == 0.5


def test_half_the_time_and_the_unfinished_one_finished_supports_both():
    r = w4_report(_evidence([_root(316, True, 7_500.0), _root(230, True, 20_000.0)]), SPEC, _w3())
    assert r["integrity"] == [] and r["time"] == {"verdict": "supported", "ratios": {316: 0.5}}
    assert r["unfinished"] == {"verdict": "supported", "finished": 1}
    assert r["sizes"][316] == {"w4": 1_000, "w3": 9_057}


def test_faster_but_not_half_is_inconclusive_and_slower_refutes():
    r = w4_report(_evidence([_root(316, True, 9_000.0), _root(230, False, 21_600.0)]), SPEC, _w3())
    assert r["time"]["verdict"] == "inconclusive" and r["unfinished"]["verdict"] == "refuted"
    assert w4_report(_evidence([_root(316, True, 16_000.0), _root(230, True, 1.0)]), SPEC, _w3())["time"][
        "verdict"] == "refuted"


def test_unfinished_where_w3_finished_refutes_the_time_claim():
    r = w4_report(_evidence([_root(316, False, 21_600.0), _root(230, True, 1.0)]), SPEC, _w3())
    assert r["time"]["verdict"] == "refuted"


def test_integrity_problems_stop_the_judgement():
    good = [_root(316, True, 1.0), _root(230, True, 1.0)]
    for broken in ("board", "shared"):
        roots = copy.deepcopy(good)
        roots[0][broken] = [0] if broken == "board" else 0
        r = w4_report(_evidence(roots), SPEC, _w3())
        assert r["integrity"] and r["time"] == {"verdict": "not_run"}
    python_walk = _evidence(copy.deepcopy(good))
    python_walk["search"] = dict(W3["search"])
    assert w4_report(python_walk, SPEC, _w3())["integrity"]
    unchecked = copy.deepcopy(good)
    unchecked[1]["moves_win"] = False
    assert w4_report(_evidence(unchecked), SPEC, _w3())["integrity"]
    foreign = _evidence(copy.deepcopy(good))
    foreign["measurement_fingerprint"] = "x"
    assert w4_report(foreign, SPEC, _w3())["integrity"]
    wrong = _evidence([_root(316, True, 1.0), _root(87, True, 1.0)])
    assert w4_report(wrong, SPEC, _w3())["integrity"]


def test_a_missing_root_is_an_integrity_problem():
    assert w4_report(_evidence([_root(316, True, 1.0)]), SPEC, _w3())["integrity"]
