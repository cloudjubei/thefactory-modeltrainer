"""Direct tests for harness/floor_a1.py — A1, four of W6's heavy frontier positions rebuilt exactly as W6 built them but
keeping a leaf with exceptions only at >= 100x under its table (W6: 30x), judged on completion and on their total size
against W6's."""
from __future__ import annotations

from harness.floor_a1 import SPEC, a1_report
from harness.floor_w6 import SPEC as W6

W6_BITS = {95: 6453, 565: 3877, 594: 32544, 535: 30009}


def _e(sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in sizes.items()]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_a1_is_w6_on_four_heavy_positions_with_only_the_accept_threshold_raised():
    assert SPEC["indices"] == [95, 565, 594, 535] and SPEC["roots"] == 4
    assert SPEC["builder"] == {**W6["builder"], "accept": 100}
    assert SPEC["search"] == W6["search"] and SPEC["w6_bits"] == W6_BITS
    assert not SPEC.get("wave_size") and not SPEC.get("library_from")


def test_completion_needs_all_four():
    assert a1_report(_e(W6_BITS), SPEC)["complete"] == {"verdict": "supported", "count": 4}
    assert a1_report(_e(W6_BITS, incomplete={95}), SPEC)["complete"]["verdict"] == "inconclusive"
    assert a1_report(_e(W6_BITS, incomplete={95, 565}), SPEC)["complete"] == {"verdict": "refuted", "count": 2}


def test_smaller_means_at_most_four_fifths_of_w6_s_total():
    total = sum(W6_BITS.values())
    fifth = {95: 0, 565: 0, 594: 0, 535: total * 4 // 5}
    r = a1_report(_e(fifth), SPEC)["size"]
    assert r == {"verdict": "supported", "bits": total * 4 // 5, "w6_bits": total, "ratio": (total * 4 // 5) / total,
                 "lower_bound": False}
    assert a1_report(_e({**fifth, 95: 1}), SPEC)["size"]["verdict"] == "inconclusive"
    exact = {**SPEC, "w6_bits": {95: 5, 565: 0, 594: 0, 535: 0}}
    assert a1_report(_e({95: 4, 565: 0, 594: 0, 535: 0}), exact)["size"]["ratio"] == 0.8
    assert a1_report(_e({95: 4, 565: 0, 594: 0, 535: 0}), exact)["size"]["verdict"] == "supported"
    assert a1_report(_e({i: n - 1 for i, n in W6_BITS.items()}), SPEC)["size"]["verdict"] == "inconclusive"
    assert a1_report(_e(W6_BITS), SPEC)["size"]["verdict"] == "refuted"


def test_an_unfinished_root_can_only_refute():
    small = {i: 10 for i in W6_BITS}
    partial = a1_report(_e(small, incomplete={594}), SPEC)["size"]
    assert partial["lower_bound"] and partial["verdict"] == "inconclusive"
    assert a1_report(_e(W6_BITS, incomplete={594}), SPEC)["size"]["verdict"] == "refuted"


def test_a_run_off_its_registration_is_not_judged():
    for bad in (_e(W6_BITS, measurement_fingerprint="other"), _e({95: 1, 565: 1, 594: 1}),
                _e(W6_BITS, builder=W6["builder"])):
        r = a1_report(bad, SPEC)
        assert r["complete"]["verdict"] == r["size"]["verdict"] == "not_run" and r["integrity"]
    wrong = _e({**{i: 1 for i in (95, 565, 594)}, 471: 1})
    assert a1_report(wrong, SPEC)["size"]["verdict"] == "not_run"
