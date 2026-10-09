"""Direct tests for harness/floor_p1.py — P1, W1's three largest frontier strategies rebuilt with long searches for
pure steady states near their roots, judged against W1 on the same positions. Synthetic evidence."""
from __future__ import annotations

import copy

from harness.floor_p1 import SPEC, p1_report
from harness.floor_w1 import SPEC as W1


def _root(index, complete, nodes, board=None):
    return {"index": index, "complete": complete, "bits": {"nodes": nodes}, "board": board or [index],
            "checked": True if complete else None, "moves_win": True if complete else None,
            "leaves_certified": True if complete else None}


W1_ROOTS = [_root(316, True, 29_935), _root(323, True, 25_324), _root(230, False, 26_062), _root(87, True, 429)]


def _evidence(roots):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": copy.deepcopy(SPEC["builder"]), "opening": {"frontier": SPEC["frontier"]}, "roots": roots}


def _w1():
    return {"roots": copy.deepcopy(W1_ROOTS)}


def test_the_registered_pilot_rebuilds_w1_s_three_largest_with_long_root_searches():
    assert SPEC["indices"] == [316, 323, 230] and SPEC["roots"] == 3 and SPEC["source"] == "opening"
    assert SPEC["builder"] == {**W1["builder"], "min_leaf_depth": 0, "seconds": 10800.0,
                               "budgets": [[0, 1200.0], [2, 300.0]]}
    assert SPEC["search"] == W1["search"]


def test_halving_the_two_strategies_w1_finished_supports():
    r = p1_report(_evidence([_root(316, True, 14_000), _root(323, True, 13_000), _root(230, False, 9_000)]), SPEC,
                  _w1())
    assert r["integrity"] == [] and r["size"]["verdict"] == "supported"
    assert r["size"]["p1"] == 27_000 and r["size"]["w1"] == 29_935 + 25_324
    assert r["complete"] == {"verdict": "inconclusive", "count": 2}


def test_no_smaller_than_w1_refutes_even_as_a_lower_bound():
    r = p1_report(_evidence([_root(316, False, 40_000), _root(323, True, 20_000), _root(230, True, 9_000)]), SPEC,
                  _w1())
    assert r["size"]["verdict"] == "refuted" and r["size"]["lower_bound"] is True


def test_an_unfinished_strategy_never_supports():
    r = p1_report(_evidence([_root(316, False, 1_000), _root(323, True, 1_000), _root(230, True, 9_000)]), SPEC,
                  _w1())
    assert r["size"]["verdict"] == "inconclusive"


def test_between_half_and_all_is_inconclusive():
    r = p1_report(_evidence([_root(316, True, 20_000), _root(323, True, 20_000), _root(230, True, 9_000)]), SPEC,
                  _w1())
    assert r["size"]["verdict"] == "inconclusive" and r["complete"]["verdict"] == "supported"


def test_exactly_half_supports_and_exactly_w1_refutes():
    half = (29_935 + 25_324) / 2
    r = p1_report(_evidence([_root(316, True, half), _root(323, True, 0), _root(230, True, 1)]), SPEC, _w1())
    assert r["size"]["verdict"] == "supported"
    r = p1_report(_evidence([_root(316, True, 29_935), _root(323, True, 25_324), _root(230, True, 1)]), SPEC, _w1())
    assert r["size"]["verdict"] == "refuted"


def test_completion_is_judged_against_its_lines():
    def run(flags):
        roots = [_root(i, f, 1_000) for i, f in zip((316, 323, 230), flags)]
        return p1_report(_evidence(roots), SPEC, _w1())["complete"]["verdict"]
    assert run((False, False, False)) == "refuted"
    assert run((True, False, False)) == "inconclusive"
    assert run((True, True, True)) == "supported"


def test_integrity_problems_stop_the_judgement():
    good = [_root(316, True, 1), _root(323, True, 1), _root(230, True, 1)]
    moved = copy.deepcopy(good)
    moved[0]["board"] = [999]
    r = p1_report(_evidence(moved), SPEC, _w1())
    assert r["integrity"] and r["size"] == {"verdict": "not_run"} and r["complete"] == {"verdict": "not_run"}
    other = _evidence(copy.deepcopy(good))
    other["builder"]["budgets"] = [[0, 60.0]]
    assert p1_report(other, SPEC, _w1())["integrity"]
    missing = _evidence(copy.deepcopy(good[:2]))
    assert p1_report(missing, SPEC, _w1())["integrity"]
    unchecked = copy.deepcopy(good)
    unchecked[1]["moves_win"] = False
    assert p1_report(_evidence(unchecked), SPEC, _w1())["integrity"]
    foreign = _evidence(copy.deepcopy(good))
    foreign["measurement_fingerprint"] = "x"
    assert p1_report(foreign, SPEC, _w1())["integrity"]
