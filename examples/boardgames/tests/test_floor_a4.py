"""Direct tests for harness/floor_a4.py — A4, eight positions (the four W7 left stuck and four heavy ones) built as W6,
improved, and then improved again with lookahead: judged on how much further lookahead cuts what plain improvement
left, overall and on the positions plain improvement could not change."""
from __future__ import annotations

from harness.floor_a4 import SPEC, a4_further, a4_unstuck
from harness.floor_w6 import SPEC as W6

POSITIONS = [95, 271, 374, 512, 594, 535, 98, 457]


def _e(rows, **over):
    roots = []
    for i, (plain, final, kept) in zip(POSITIONS, rows):
        roots.append({"index": i, "complete": plain is not None, "bits": {"nodes": final}, "build_seconds": 600.0,
                      "checked": True, "moves_win": True, "leaves_certified": True,
                      "improved": None if plain is None else {"plain_bits": plain, "kept": kept,
                                                              "lookahead": {"tried": 1, "kept": 0}}})
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_a4_is_w6_improved_then_improved_with_lookahead_on_eight_positions():
    assert SPEC["indices"] == POSITIONS and SPEC["roots"] == 8
    assert SPEC["builder"] == {**W6["builder"], "improve": True, "lookahead": True}
    assert SPEC["search"] == W6["search"] and SPEC["measurement_fp"] == "0594eb30c588"


def test_the_further_cut_is_judged_on_the_total_plain_improvement_left():
    rows = [(1000, 500, 1)] + [(1000, 1000, 1)] * 7
    r = a4_further(_e([(8000, 6800, 1)] + [(0, 0, 1)] * 7), SPEC)
    assert r == {"verdict": "supported", "plain_bits": 8000, "bits": 6800, "cut": 0.15}
    assert a4_further(_e(rows), SPEC)["verdict"] == "inconclusive"
    assert a4_further(_e([(1000, 970, 1)] + [(0, 0, 1)] * 7), SPEC)["verdict"] == "inconclusive"
    assert a4_further(_e([(1000, 971, 1)] + [(0, 0, 1)] * 7), SPEC)["verdict"] == "refuted"
    assert a4_further(_e([(1000, 1000, 1)] * 8), SPEC)["cut"] == 0


def test_a_position_that_did_not_finish_leaves_the_cut_undecided():
    rows = [(None, 5000, 0)] + [(1000, 100, 1)] * 7
    assert a4_further(_e(rows), SPEC)["verdict"] == "inconclusive"
    assert a4_unstuck(_e(rows), SPEC)["verdict"] == "inconclusive"


def test_unstuck_is_judged_on_the_heavy_positions_plain_improvement_could_not_change():
    stuck = [(1000, 500, 0), (1000, 1000, 0), (150, 100, 0), (9000, 100, 2)] + [(1000, 1000, 1)] * 4
    r = a4_unstuck(_e(stuck), SPEC)
    assert r == {"verdict": "supported", "stuck": 2, "cut": 1}
    none = [(1000, 1000, 0), (1000, 1000, 0)] + [(1000, 1000, 1)] * 6
    assert a4_unstuck(_e(none), SPEC)["verdict"] == "refuted"
    three = [(1000, 900, 0), (1000, 1000, 0), (1000, 1000, 0)] + [(1000, 1000, 1)] * 5
    assert a4_unstuck(_e(three), SPEC) == {"verdict": "inconclusive", "stuck": 3, "cut": 1}
    one = [(1000, 1000, 0)] + [(1000, 1000, 1)] * 7
    assert a4_unstuck(_e(one), SPEC) == {"verdict": "inconclusive", "stuck": 1, "cut": 0}


def test_a_run_off_its_registration_is_not_judged():
    rows = [(1000, 100, 1)] * 8
    for bad in (_e(rows, measurement_fingerprint=W6["measurement_fp"]), _e(rows[:7]), _e(rows, builder=W6["builder"])):
        assert a4_further(bad, SPEC)["verdict"] == a4_unstuck(bad, SPEC)["verdict"] == "not_run"
    moved = _e(rows)
    moved["roots"][0]["index"] = 471
    assert a4_further(moved, SPEC)["verdict"] == "not_run"
