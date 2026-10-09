"""Direct tests for harness/floor_a2.py — A2, A1's four heavy positions rebuilt as W6 built them (leaves with exceptions
from 30x) with the builder choosing between such a leaf and the split below it by size, judged on completion and on
their total against the smaller build per position of W6 and A1 (28,248 bits) and against A1's 42,464."""
from __future__ import annotations

from harness.floor_a1 import SPEC as A1
from harness.floor_a2 import SPEC, a2_report
from harness.floor_w6 import SPEC as W6

POSITIONS = (95, 565, 594, 535)


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(POSITIONS, sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_a2_is_w6_on_a1_s_positions_choosing_by_size():
    assert SPEC["indices"] == A1["indices"] and SPEC["roots"] == 4
    assert SPEC["builder"] == {**W6["builder"], "choose_by_size": True}
    assert SPEC["search"] == W6["search"] and SPEC["measurement_fp"] != W6["measurement_fp"]
    assert (SPEC["best_bits"], SPEC["a1_bits"]) == (28_248, 42_464)
    assert not SPEC.get("wave_size") and not SPEC.get("library_from")


def test_completion_needs_all_four():
    assert a2_report(_e([1, 1, 1, 1]), SPEC)["complete"] == {"verdict": "supported", "count": 4}
    assert a2_report(_e([1, 1, 1, 1], incomplete={594}), SPEC)["complete"]["verdict"] == "inconclusive"
    assert a2_report(_e([1, 1, 1, 1], incomplete={594, 535}), SPEC)["complete"] == {"verdict": "refuted",
                                                                                  "count": 2}


def test_the_size_is_judged_against_the_best_per_position_and_against_a1():
    at_best = a2_report(_e([28_248, 0, 0, 0]), SPEC)["size"]
    assert at_best == {"verdict": "supported", "bits": 28_248, "lower_bound": False}
    assert a2_report(_e([28_249, 0, 0, 0]), SPEC)["size"]["verdict"] == "inconclusive"
    assert a2_report(_e([42_463, 0, 0, 0]), SPEC)["size"]["verdict"] == "inconclusive"
    assert a2_report(_e([42_464, 0, 0, 0]), SPEC)["size"]["verdict"] == "refuted"
    assert a2_report(_e([7_000, 7_000, 7_000, 7_000]), SPEC)["size"]["bits"] == 28_000


def test_an_unfinished_root_can_only_refute():
    partial = a2_report(_e([10, 10, 10, 10], incomplete={95}), SPEC)["size"]
    assert partial["lower_bound"] and partial["verdict"] == "inconclusive"
    assert a2_report(_e([42_464, 0, 0, 0], incomplete={95}), SPEC)["size"]["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([1, 1, 1, 1], measurement_fingerprint=W6["measurement_fp"]), _e([1, 1, 1]),
                _e([1, 1, 1, 1], builder=W6["builder"])):
        r = a2_report(bad, SPEC)
        assert r["integrity"] and r["complete"]["verdict"] == r["size"]["verdict"] == "not_run"
    moved = _e([1, 1, 1, 1])
    moved["roots"][0]["index"] = 471
    assert a2_report(moved, SPEC)["size"]["verdict"] == "not_run"
