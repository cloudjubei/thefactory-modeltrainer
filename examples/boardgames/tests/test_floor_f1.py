"""Direct tests for harness/floor_f1.py — F1, every frontier position built with L2's rule and pipeline in waves from
a growing library: judged on completion, on the whole strategy's size against W9's projection, and against the net."""
from __future__ import annotations

from harness.floor_f1 import SPEC, f1_bits, f1_complete, f1_net, f1_size
from harness.floor_w9 import SPEC as W9
from harness.floor_w10 import SPEC as W10

BASE = 204 * 4 + 259 + 8_618


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(SPEC["indices"], sizes)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_f1_builds_every_frontier_position_in_w1_s_order_with_l2_s_rule_and_a_growing_library():
    assert sorted(SPEC["indices"]) == list(range(671)) and SPEC["roots"] == 671
    assert SPEC["indices"][:8] == [362, 87, 316, 346, 230, 323, 393, 263]
    assert SPEC["indices"][8:11] == [471, 178, 95] and SPEC["indices"][40:43] == W9["indices"][:3]
    assert SPEC["builder"] == W10["builder"] and SPEC["builder"]["ties"] == "lowest"
    assert "safe_depth" not in SPEC["builder"] and SPEC["search"] == W10["search"]
    assert (SPEC["wave_size"], SPEC["library_from"], SPEC["library_bits"]) == (32, "c49_l2.json.gz", 8_618)
    assert SPEC["measurement_fp"] == "4242d4bca6e2"
    assert (SPEC["w9_bits"], SPEC["w9_high"], SPEC["net_bits"], SPEC["unfinished_refute"]) == (
        271_740, 476_290, 660_000, 7)


def test_the_size_is_the_opening_the_seed_library_and_every_position():
    assert f1_bits(_e([100] * 671), SPEC) == BASE + 67_100


def test_completion_needs_every_position():
    assert f1_complete(_e([1] * 671), SPEC) == {"verdict": "supported", "unfinished": 0}
    assert f1_complete(_e([1] * 671, incomplete={362}), SPEC) == {"verdict": "inconclusive", "unfinished": 1}
    some = set(SPEC["indices"][:7])
    assert f1_complete(_e([1] * 671, incomplete=some), SPEC) == {"verdict": "inconclusive", "unfinished": 7}
    assert f1_complete(_e([1] * 671, incomplete=set(SPEC["indices"][:8])), SPEC)["verdict"] == "refuted"


def test_the_size_is_judged_against_w9_s_projection_and_its_high():
    at = (271_740 - BASE) // 671
    assert f1_size(_e([at] * 671), SPEC)["verdict"] == "supported"
    exact = {**SPEC, "w9_bits": BASE + 671 * at}
    assert f1_size(_e([at] * 671), exact)["verdict"] == "supported"
    over = {**SPEC, "w9_bits": BASE + 671 * at - 1}
    assert f1_size(_e([at] * 671), over)["verdict"] == "inconclusive"
    high = {**SPEC, "w9_high": BASE + 671 * at}
    assert f1_size(_e([at] * 671), high)["verdict"] == "supported"
    assert f1_size(_e([at] * 670 + [at + 1]), {**high, "w9_bits": 0})["verdict"] == "refuted"
    partial = f1_size(_e([1] * 671, incomplete={362}), SPEC)
    assert partial["lower_bound"] and partial["verdict"] == "inconclusive"
    assert f1_size(_e([10_000] * 671, incomplete={362}), SPEC)["verdict"] == "refuted"


def test_below_the_net_only_when_complete_and_above_refutes_even_unfinished():
    assert f1_net(_e([100] * 671), SPEC)["verdict"] == "supported"
    assert f1_net(_e([100] * 671, incomplete={362}), SPEC)["verdict"] == "inconclusive"
    at = {**SPEC, "net_bits": BASE + 67_100}
    assert f1_net(_e([100] * 671), at)["verdict"] == "refuted"
    assert f1_net(_e([100] * 671, incomplete={362}), at)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    sizes = [1] * 671
    moved = _e(sizes)
    moved["roots"][0]["index"] = moved["roots"][1]["index"]
    for bad in (_e(sizes, measurement_fingerprint="other"), _e(sizes[:670]), _e(sizes, builder=W9["builder"] | {
            "ties": "exact"}), moved):
        assert f1_complete(bad, SPEC)["verdict"] == f1_size(bad, SPEC)["verdict"] == f1_net(bad, SPEC)["verdict"] == (
            "not_run")
