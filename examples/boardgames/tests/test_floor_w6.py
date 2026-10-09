"""Direct tests for harness/floor_w6.py — W6, the next 32 of W1's seeded frontier positions built exactly as W5b
builds (no library: W2 and W5b used it for 2 of 61 and 0 of their leaves), judged on completion, on whether the whole
game projects above the trained net (a bootstrap interval of the sample's mean), and on how much of the sample's bits
its 4 largest strategies carry."""
from __future__ import annotations

import pytest

from harness.floor_w1 import SPEC as W1
from harness.floor_w5b import SPEC as W5B
from harness.floor_w6 import SPEC, w6_projection, w6_report, w6_tail

OPENING = {"moves": 204, "trivial": 259, "frontier": 671}
OPENING_BITS = 204 * 4 + 259


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in enumerate(sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": OPENING, "roots": roots, **over}


def _mean_for(bits):
    return (bits - OPENING_BITS) / OPENING["frontier"]


def test_w6_builds_the_next_32_of_w1_s_order_exactly_as_w5b_without_a_library():
    assert SPEC["builder"] == W5B["builder"] and SPEC["search"] == W5B["search"]
    assert SPEC["seed"] == W1["seed"] and SPEC["source"] == "opening" and SPEC["ply"] == 8
    assert len(SPEC["indices"]) == SPEC["roots"] == 32 and len(set(SPEC["indices"])) == 32
    assert not set(SPEC["indices"]) & set(W5B["indices"])
    assert SPEC["indices"][:3] == [471, 178, 95] and SPEC["indices"][-1] == 457
    assert not SPEC.get("wave_size") and not SPEC.get("library_from")


def test_completion_is_judged_on_the_32():
    sizes = [100] * 32
    assert w6_report(_e(sizes, incomplete=range(4)), SPEC)["complete"] == {"verdict": "supported", "count": 28}
    assert w6_report(_e(sizes, incomplete=range(5)), SPEC)["complete"]["verdict"] == "inconclusive"
    assert w6_report(_e(sizes, incomplete=range(7)), SPEC)["complete"]["verdict"] == "inconclusive"
    assert w6_report(_e(sizes, incomplete=range(8)), SPEC)["complete"] == {"verdict": "refuted", "count": 24}


def test_a_run_off_its_registration_is_not_judged():
    sizes = [5000] * 32
    for bad in (_e(sizes, measurement_fingerprint="other"), _e(sizes[:31]),
                _e(sizes, builder={**SPEC["builder"], "seconds": 1.0}),
                _e(sizes, opening={**OPENING, "frontier": 670})):
        assert w6_report(bad, SPEC)["complete"]["verdict"] == "not_run"
        assert w6_projection(bad, SPEC)["verdict"] == "not_run"
        assert w6_tail(bad, SPEC)["verdict"] == "not_run"
    failed = _e(sizes)
    failed["roots"][3]["leaves_certified"] = False
    assert w6_projection(failed, SPEC)["verdict"] == "not_run"


def test_the_projection_is_the_opening_plus_the_frontier_times_the_mean_with_a_bootstrap_interval():
    r = w6_projection(_e([1000] * 32), SPEC)
    assert r["bits"] == r["low"] == r["high"] == OPENING_BITS + 671 * 1000
    assert r["verdict"] == "supported" and r["hours"] == pytest.approx(671 * 600 / 3600)
    spread = w6_projection(_e([200] * 16 + [3000] * 16), SPEC)
    assert spread["low"] < spread["bits"] < spread["high"]
    assert spread["bits"] == OPENING_BITS + 671 * 1600


def test_the_projection_is_above_the_net_only_when_the_whole_interval_is():
    above = _mean_for(SPEC["net_bits"]) + 1
    assert w6_projection(_e([above] * 32), SPEC)["verdict"] == "supported"
    at_the_net = {**SPEC, "net_bits": OPENING_BITS + 671 * 1000}
    assert w6_projection(_e([1000] * 32), at_the_net)["verdict"] == "inconclusive"
    tail = w6_projection(_e([300] * 31 + [40_000]), SPEC)
    assert tail["bits"] > SPEC["net_bits"] > tail["low"] and tail["verdict"] == "inconclusive"


def test_the_projection_is_below_the_net_only_when_every_root_finished():
    below = [300] * 32
    assert w6_projection(_e(below), SPEC)["verdict"] == "refuted"
    straddling = w6_projection(_e([300] * 31 + [10_000]), SPEC)
    assert straddling["bits"] < SPEC["net_bits"] < straddling["high"] and straddling["verdict"] == "inconclusive"
    partial = w6_projection(_e(below, incomplete={7}), SPEC)
    assert partial["lower_bound"] and partial["verdict"] == "inconclusive"
    assert w6_projection(_e([2000] * 32, incomplete={7}), SPEC)["verdict"] == "supported"


def test_the_tail_is_the_share_of_the_bits_in_the_4_largest():
    r = w6_tail(_e([100] * 28 + [700] * 4), SPEC)
    assert r == {"verdict": "supported", "share": 0.5, "largest": [700, 700, 700, 700]}
    assert w6_tail(_e([700] + [100] * 28 + [700] * 3), SPEC)["share"] == 0.5
    assert w6_tail(_e([100] * 28 + [300] * 4), SPEC)["verdict"] == "inconclusive"
    assert w6_tail(_e([100] * 28 + [180] * 4), SPEC)["verdict"] == "refuted"
    quarter = w6_tail(_e([30] * 28 + [70] * 4), SPEC)
    assert quarter["share"] == 0.25 and quarter["verdict"] == "inconclusive"
    assert w6_tail(_e([100] * 32), SPEC)["verdict"] == "refuted"


def test_the_tail_is_not_judged_while_any_root_is_unfinished():
    assert w6_tail(_e([100] * 28 + [700] * 4, incomplete={0}), SPEC)["verdict"] == "inconclusive"
