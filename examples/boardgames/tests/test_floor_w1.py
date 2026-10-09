"""Direct tests for harness/floor_w1.py — W1, the whole-game projection: an opening to ply 8 plus a sample of the
strategies its frontier needs, judged on completion and on the projected size of the whole first-player strategy
(a lower bound while any sampled root is unfinished). Synthetic evidence."""
from __future__ import annotations

import copy

from harness.floor_w1 import SPEC, w1_report


def _root(index, complete, nodes, seconds=100.0):
    return {"index": index, "complete": complete, "bits": {"nodes": nodes}, "build_seconds": seconds,
            "checked": True if complete else None, "moves_win": True if complete else None,
            "leaves_certified": True if complete else None}


def _evidence(roots, frontier=671, moves=204, trivial=259):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": copy.deepcopy(SPEC["builder"]), "opening": {"moves": moves, "trivial": trivial,
                                                                     "frontier": frontier}, "roots": roots}


def test_the_registered_study_samples_8_frontier_roots_of_the_ply_8_opening_with_s5_s_leaves_for_2_hours():
    assert (SPEC["ply"], SPEC["roots"], SPEC["seed"], SPEC["source"], SPEC["frontier"]) == (8, 8, 4, "opening", 671)
    assert SPEC["builder"] == {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 2,
                               "reuse_window": 200, "seconds": 7200.0, "accept": 30}
    assert (SPEC["projection_support"], SPEC["projection_refute"]) == (660_000, 6_600_000)


def test_all_complete_and_small_projects_under_the_net():
    r = w1_report(_evidence([_root(i, True, 900) for i in range(8)]), SPEC)
    assert r["integrity"] == [] and r["complete"] == {"verdict": "supported", "count": 8}
    assert r["projection"]["bits"] == 204 * 4 + 259 + 671 * 900 and r["projection"]["verdict"] == "supported"
    assert r["projection"]["lower_bound"] is False


def test_a_lower_bound_above_the_refute_line_refutes_even_with_unfinished_roots():
    roots = [_root(i, i < 2, 20_000) for i in range(8)]
    r = w1_report(_evidence(roots), SPEC)
    assert r["complete"]["verdict"] == "refuted" and r["projection"]["lower_bound"] is True
    assert r["projection"]["verdict"] == "refuted"


def test_an_unfinished_root_keeps_a_small_projection_inconclusive():
    roots = [_root(i, i < 7, 900) for i in range(8)]
    r = w1_report(_evidence(roots), SPEC)
    assert r["complete"]["verdict"] == "supported" and r["projection"]["verdict"] == "inconclusive"


def test_a_projection_between_the_lines_is_inconclusive():
    r = w1_report(_evidence([_root(i, True, 3_000) for i in range(8)]), SPEC)
    assert r["projection"]["verdict"] == "inconclusive"


def test_completion_is_judged_against_its_lines():
    assert w1_report(_evidence([_root(i, i < 3, 900) for i in range(8)]), SPEC)["complete"]["verdict"] == "refuted"
    assert w1_report(_evidence([_root(i, i < 4, 900) for i in range(8)]), SPEC)["complete"]["verdict"] == \
        "inconclusive"
    assert w1_report(_evidence([_root(i, i < 6, 900) for i in range(8)]), SPEC)["complete"]["verdict"] == \
        "supported"


def test_the_build_time_is_projected_over_the_frontier():
    r = w1_report(_evidence([_root(i, True, 900, seconds=360.0) for i in range(8)]), SPEC)
    assert r["projection"]["hours"] == 671 * 360.0 / 3600


def test_integrity_problems_stop_the_judgement():
    bad = _evidence([_root(i, True, 900) for i in range(8)], frontier=670)
    r = w1_report(bad, SPEC)
    assert r["integrity"] and r["complete"] == {"verdict": "not_run"} and r["projection"] == {"verdict": "not_run"}
    other = _evidence([_root(i, True, 900) for i in range(8)])
    other["builder"]["seconds"] = 60.0
    assert w1_report(other, SPEC)["integrity"]
    wrong = _evidence([_root(i, True, 900) for i in range(7)])
    assert w1_report(wrong, SPEC)["integrity"]
    unchecked = _evidence([_root(i, True, 900) for i in range(8)])
    unchecked["roots"][0]["leaves_certified"] = False
    assert w1_report(unchecked, SPEC)["integrity"]
    foreign = _evidence([_root(i, True, 900) for i in range(8)])
    foreign["measurement_fingerprint"] = "x"
    assert w1_report(foreign, SPEC)["integrity"]


def test_an_unfinished_root_counts_at_its_partial_size():
    roots = [_root(i, True, 900) for i in range(7)] + [_root(7, False, 9_000)]
    r = w1_report(_evidence(roots), SPEC)
    assert r["projection"]["bits"] == 204 * 4 + 259 + 671 * (7 * 900 + 9_000) / 8
