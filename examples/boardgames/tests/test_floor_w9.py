"""Direct tests for harness/floor_w9.py — W9, the next 32 frontier positions of W1's seeded order (never built) with
L2's rule and pipeline, judged on completion as W6 and on whether the whole game projects below the trained net."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_w6 import SPEC as W6, w6_report
from harness.floor_w9 import SPEC, w9_below, w9_report

BASE = 204 * 4 + 259


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(SPEC["indices"], sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_w9_is_l2_on_the_next_32_fresh_positions_of_w1_s_order():
    assert SPEC["builder"] == L2["builder"] and SPEC["builder"]["ties"] == "lowest"
    assert SPEC["search"] == L2["search"] and SPEC["measurement_fp"] == L2["measurement_fp"]
    assert len(SPEC["indices"]) == SPEC["roots"] == 32 and len(set(SPEC["indices"])) == 32
    assert not set(SPEC["indices"]) & (set(W6["indices"]) | {362, 87, 316, 346, 230, 323, 393, 263})
    assert SPEC["indices"][:3] == [615, 180, 504] and SPEC["indices"][-1] == 208
    assert (SPEC["net_bits"], SPEC["bootstrap"], SPEC["bootstrap_seed"]) == (660_000, 10_000, 6)
    assert w9_report is w6_report


def test_below_the_net_needs_the_whole_interval_below_and_every_root_finished():
    under = (660_000 - BASE) // 671 - 1
    r = w9_below(_e([under] * 32), SPEC)
    assert r["verdict"] == "supported" and r["high"] < 660_000
    assert w9_below(_e([under] * 32, incomplete={615}), SPEC)["verdict"] == "inconclusive"
    tail = w9_below(_e([300] * 31 + [40_000]), SPEC)
    assert tail["low"] < 660_000 < tail["high"] and tail["verdict"] == "inconclusive"


def test_above_the_net_refutes_even_when_unfinished():
    over = (660_000 - BASE) // 671 + 2
    assert w9_below(_e([over] * 32), SPEC)["verdict"] == "refuted"
    assert w9_below(_e([over] * 32, incomplete={615}), SPEC)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([10] * 32, measurement_fingerprint="other"), _e([10] * 31), _e([10] * 32, builder=W6["builder"])):
        assert w9_below(bad, SPEC)["verdict"] == "not_run"
