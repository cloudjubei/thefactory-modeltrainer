"""Direct tests for harness/floor_w10.py — W10, W9's 32 fresh positions rebuilt with L2's rule and pipeline from a
library of the 90 maps L2 found on W6's positions, judged on completion as W6, on the whole game's projection with the
library's bits paid once against W9's, and on how many leaves use the library."""
from __future__ import annotations

from harness.floor_w6 import w6_report
from harness.floor_w9 import SPEC as W9
from harness.floor_w10 import SPEC, w10_library, w10_projection, w10_report

BASE = 204 * 4 + 259


def _e(sizes, incomplete=(), leaves=None, used=None, **over):
    leaves = leaves or [1] * 32
    used = used or [0] * 32
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s, "leaves": lv}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True, "shared": 90, "library_leaves": u}
             for i, s, lv, u in zip(SPEC["indices"], sizes, leaves, used)]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_w10_is_w9_built_from_l2_s_maps():
    assert SPEC["indices"] == W9["indices"] and SPEC["builder"] == W9["builder"] and SPEC["search"] == W9["search"]
    assert SPEC["measurement_fp"] == "081011f9bbfa" != W9["measurement_fp"] and SPEC["library_from"] == "c49_l2.json.gz"
    assert (SPEC["library_bits"], SPEC["w9_bits"], SPEC["support"]) == (8_618, 271_740, 0.8)
    assert (SPEC["library_support"], SPEC["library_refute"]) == (0.2, 0.05)
    assert w10_report is w6_report


def test_the_projection_pays_the_library_once_and_is_judged_against_w9():
    spec = {**SPEC, "w9_bits": (BASE + 8_618 + 671 * 1000) / 0.8}
    r = w10_projection(_e([1000] * 32), spec)
    assert r["verdict"] == "supported" and r["bits"] == BASE + 8_618 + 671 * 1000 and not r["lower_bound"]
    assert w10_projection(_e([1000] * 31 + [1001]), spec)["verdict"] == "inconclusive"
    at_w9 = {**SPEC, "w9_bits": BASE + 8_618 + 671 * 1000}
    assert w10_projection(_e([1000] * 31 + [999]), at_w9)["verdict"] == "inconclusive"
    assert w10_projection(_e([1000] * 32), at_w9)["verdict"] == "refuted"
    assert w10_projection(_e([1000] * 32, incomplete={615}), at_w9)["verdict"] == "refuted"
    assert w10_projection(_e([10] * 32, incomplete={615}), SPEC)["verdict"] == "inconclusive"


def test_library_use_is_the_share_of_leaves_that_reference_it():
    r = w10_library(_e([10] * 32, leaves=[5] * 32, used=[1] * 32), SPEC)
    assert r == {"verdict": "supported", "share": 0.2, "used": 32, "leaves": 160}
    assert w10_library(_e([10] * 32, leaves=[5] * 32, used=[1] * 31 + [0]), SPEC)["verdict"] == "inconclusive"
    assert w10_library(_e([10] * 32, leaves=[20] * 32, used=[1] * 32), SPEC)["verdict"] == "inconclusive"
    assert w10_library(_e([10] * 32, leaves=[20] * 32, used=[1] * 31 + [0]), SPEC)["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e([10] * 32, measurement_fingerprint="other"), _e([10] * 31), _e([10] * 32, builder=W9["builder"] |
                                                                                    {"ties": "exact"})):
        assert w10_projection(bad, SPEC)["verdict"] == w10_library(bad, SPEC)["verdict"] == "not_run"
