"""Direct tests for harness/floor_w8.py — W8, W6's 32 frontier positions built, improved and improved with lookahead:
judged on completion and against the net as W6, on the projection against W7's 1.47M bits, and on lookahead's cut
measured inside the 32 builds."""
from __future__ import annotations

from harness.floor_w6 import w6_projection, w6_report
from harness.floor_w7 import SPEC as W7
from harness.floor_w8 import SPEC, w8_lookahead, w8_projection, w8_report, w8_vs_w7

BASE = 204 * 4 + 259


def _e(sizes, plain=None, incomplete=(), **over):
    plain = plain or sizes
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True,
              "improved": None if i in incomplete else {"plain_bits": p}}
             for i, s, p in zip(SPEC["indices"], sizes, plain)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_w8_is_w7_with_lookahead_on_the_same_positions():
    assert SPEC["indices"] == W7["indices"] and SPEC["roots"] == 32
    assert SPEC["builder"] == {**W7["builder"], "lookahead": True} and SPEC["search"] == W7["search"]
    assert SPEC["measurement_fp"] == "b2f5efcf2ac9" and SPEC["w7_bits"] == 1_465_679
    assert w8_report is w6_report and w8_projection is w6_projection


def test_the_projection_is_judged_against_w7_s():
    bits = BASE + 671 * 1000
    spec = {**SPEC, "w7_bits": bits * 10 / 9}
    r = w8_vs_w7(_e([1000] * 32), spec)
    assert r["verdict"] == "supported" and r["bits"] == bits and not r["lower_bound"]
    assert w8_vs_w7(_e([1000] * 31 + [1001]), spec)["verdict"] == "inconclusive"
    at_w7 = {**SPEC, "w7_bits": bits}
    assert w8_vs_w7(_e([1000] * 31 + [999]), at_w7)["verdict"] == "inconclusive"
    assert w8_vs_w7(_e([1000] * 32), at_w7)["verdict"] == "refuted"
    assert w8_vs_w7(_e([1000] * 32, incomplete={471}), at_w7)["verdict"] == "refuted"
    assert w8_vs_w7(_e([10] * 32, incomplete={471}), SPEC)["verdict"] == "inconclusive"


def test_lookahead_s_cut_is_measured_inside_the_builds():
    r = w8_lookahead(_e([900] + [0] * 31, plain=[1000] + [0] * 31), SPEC)
    assert r == {"verdict": "supported", "plain_bits": 1000, "bits": 900, "cut": 0.1}
    assert w8_lookahead(_e([901] + [0] * 31, plain=[1000] + [0] * 31), SPEC)["verdict"] == "inconclusive"
    assert w8_lookahead(_e([980] + [0] * 31, plain=[1000] + [0] * 31), SPEC)["verdict"] == "inconclusive"
    assert w8_lookahead(_e([981] + [0] * 31, plain=[1000] + [0] * 31), SPEC)["verdict"] == "refuted"
    assert w8_lookahead(_e([500, 500] + [0] * 30, plain=[1000, 500] + [0] * 30), SPEC)["cut"] == 0.5 / 1.5
    assert w8_lookahead(_e([1] * 32, incomplete={471}), SPEC)["verdict"] == "inconclusive"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([10] * 32, measurement_fingerprint=W7["measurement_fp"]), _e([10] * 31),
                _e([10] * 32, builder=W7["builder"])):
        assert w8_vs_w7(bad, SPEC)["verdict"] == w8_lookahead(bad, SPEC)["verdict"] == "not_run"
