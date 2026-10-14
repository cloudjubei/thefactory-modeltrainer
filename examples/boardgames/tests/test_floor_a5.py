"""Direct tests for harness/floor_a5.py — A5, A4's eight positions built, improved and improved with lookahead as A4,
then their costliest leaves re-searched for 120 s each in the time left and improved with lookahead once more: judged
on how much further that cuts what the lookahead pass left."""
from __future__ import annotations

from harness.floor_a4 import SPEC as A4
from harness.floor_a5 import SPEC, a5_further

POSITIONS = [95, 271, 374, 512, 594, 535, 98, 457]


def _e(rows, **over):
    roots = [{"index": i, "complete": before is not None, "bits": {"nodes": final}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True,
              "improved": None if before is None else {"lookahead_bits": before}}
             for i, (before, final) in zip(POSITIONS, rows)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_a5_is_a4_plus_research_on_the_same_positions():
    assert SPEC["indices"] == A4["indices"] == POSITIONS and SPEC["roots"] == 8
    assert SPEC["builder"] == {**A4["builder"], "research": 120.0} and SPEC["search"] == A4["search"]
    assert SPEC["measurement_fp"] == "b2f5efcf2ac9" != A4["measurement_fp"]


def test_the_further_cut_is_judged_on_what_the_lookahead_pass_left():
    r = a5_further(_e([(8000, 7200)] + [(0, 0)] * 7), SPEC)
    assert r == {"verdict": "supported", "lookahead_bits": 8000, "bits": 7200, "cut": 0.1}
    assert a5_further(_e([(1000, 980)] + [(0, 0)] * 7), SPEC)["verdict"] == "inconclusive"
    assert a5_further(_e([(1000, 981)] + [(0, 0)] * 7), SPEC)["verdict"] == "refuted"
    assert a5_further(_e([(1000, 901)] + [(0, 0)] * 7), SPEC)["verdict"] == "inconclusive"
    assert a5_further(_e([(500, 400), (500, 500)] + [(0, 0)] * 6), SPEC)["cut"] == 0.1


def test_an_unfinished_position_leaves_the_cut_undecided():
    assert a5_further(_e([(None, 900)] + [(1000, 100)] * 7), SPEC)["verdict"] == "inconclusive"


def test_a_run_off_its_registration_is_not_judged():
    rows = [(1000, 100)] * 8
    for bad in (_e(rows, measurement_fingerprint=A4["measurement_fp"]), _e(rows[:7]), _e(rows, builder=A4["builder"])):
        assert a5_further(bad, SPEC)["verdict"] == "not_run"
    moved = _e(rows)
    moved["roots"][0]["index"] = 471
    assert a5_further(moved, SPEC)["verdict"] == "not_run"
