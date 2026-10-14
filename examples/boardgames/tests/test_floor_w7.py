"""Direct tests for harness/floor_w7.py — W7, W6's 32 frontier positions rebuilt exactly as W6 built them plus the
builder's anytime improvement, judged on completion and against the net as W6 (harness.floor_w6) and on the projection
against W6's own 4.31M bits."""
from __future__ import annotations

from harness.floor_w6 import SPEC as W6, w6_projection, w6_report
from harness.floor_w7 import SPEC, w7_halving, w7_projection, w7_report

OPENING = {"moves": 204, "trivial": 259, "frontier": 671}
BASE = 204 * 4 + 259


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(SPEC["indices"], sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": OPENING, "roots": roots, **over}


def test_w7_is_w6_on_the_same_positions_plus_improvement():
    assert SPEC["indices"] == W6["indices"] and SPEC["roots"] == 32
    assert SPEC["builder"] == {**W6["builder"], "improve": True} and SPEC["search"] == W6["search"]
    assert SPEC["measurement_fp"] == "79d96bd72825" and SPEC["w6_bits"] == 4_306_421
    assert w7_report is w6_report and w7_projection is w6_projection


def test_halving_is_judged_against_w6_s_projection():
    bits = BASE + 671 * 1000
    spec = {**SPEC, "w6_bits": 2 * bits}
    r = w7_halving(_e([1000] * 32), spec)
    assert r == {"verdict": "supported", "bits": bits, "ratio": 0.5, "lower_bound": False}
    assert w7_halving(_e([1000] * 31 + [1001]), spec)["verdict"] == "inconclusive"
    assert w7_halving(_e([2000] * 31 + [1999]), spec)["verdict"] == "inconclusive"
    assert w7_halving(_e([2000] * 32), {**SPEC, "w6_bits": BASE + 671 * 2000})["verdict"] == "refuted"
    assert w7_halving(_e([1000] * 16 + [3000] * 16), spec)["ratio"] == (BASE + 671 * 2000) / (2 * bits)


def test_an_unfinished_root_can_only_refute_the_halving():
    assert w7_halving(_e([10] * 32, incomplete={471}), SPEC)["verdict"] == "inconclusive"
    at_w6 = {**SPEC, "w6_bits": BASE + 671 * 2000}
    assert w7_halving(_e([2000] * 32, incomplete={471}), at_w6)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([10] * 32, measurement_fingerprint=W6["measurement_fp"]), _e([10] * 31),
                _e([10] * 32, builder=W6["builder"])):
        assert w7_halving(bad, SPEC)["verdict"] == "not_run"
        assert w7_report(bad, SPEC)["complete"]["verdict"] == "not_run"
