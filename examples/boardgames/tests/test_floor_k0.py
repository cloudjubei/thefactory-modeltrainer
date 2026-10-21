"""Direct tests for harness/floor_k0.py — K0, the pilot before K1: the same leaf search without and with two-phase maps
at positions L6's heavy strategies split, judged by sign over the positions where the two differ."""
from __future__ import annotations

from harness.floor_k0 import SPEC, k0_sign
from harness.floor_l4 import SPEC as L4

ROOTS = [i for i in L4["indices"][:16] if i != 297]


def _row(root, one, two, switched=False):
    return {"root": root, "one": {"bits": one, "switched": False}, "two": {"bits": two, "switched": switched}}


def _e(pairs, **over):
    rows = [_row(ROOTS[i % len(ROOTS)], one, two, switched) for i, (one, two, switched) in enumerate(pairs)]
    rows += [_row(r, 5, 5) for r in ROOTS]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "source": SPEC["source"], "rows": rows, **over}


def test_k0_searches_l6_s_heavy_splits():
    assert SPEC["source"] == "c49_l6.json.gz" and SPEC["roots"] == L4["indices"][:16]
    assert (SPEC["per_root"], SPEC["seed"], SPEC["seconds"], SPEC["cap"]) == (3, 0, 30.0, 1_000_000)
    assert (SPEC["support"], SPEC["min_differ"]) == (2, 10)


def test_twice_as_often_smaller_supports_and_no_more_often_refutes():
    assert k0_sign(_e([(10, 9, True)] * 8 + [(9, 10, False)] * 4), SPEC) == {
        "verdict": "supported", "smaller": 8, "larger": 4, "switched": 8}
    assert k0_sign(_e([(10, 9, False)] * 7 + [(9, 10, False)] * 4), SPEC)["verdict"] == "inconclusive"
    assert k0_sign(_e([(10, 9, False)] * 6 + [(9, 10, False)] * 6), SPEC)["verdict"] == "refuted"


def test_too_few_differing_positions_cannot_judge():
    assert k0_sign(_e([(10, 9, False)] * 10), SPEC)["verdict"] == "supported"
    assert k0_sign(_e([(10, 9, False)] * 9), SPEC)["verdict"] == "inconclusive"
    assert k0_sign(_e([(9, 10, False)] * 9), SPEC)["verdict"] == "inconclusive"


def test_a_search_with_no_leaf_is_larger_than_any():
    assert k0_sign(_e([(None, 900, False)] * 10), SPEC)["smaller"] == 10
    assert k0_sign(_e([(900, None, False)] * 10), SPEC)["larger"] == 10
    assert k0_sign(_e([(None, None, False)] * 10), SPEC)["verdict"] == "inconclusive"


def test_another_measurement_is_not_judged():
    good = [(10, 9, False)] * 10
    assert k0_sign(_e(good, measurement_fingerprint="x"), SPEC)["verdict"] == "not_run"
    assert k0_sign(_e(good, source="c49_l5.json.gz"), SPEC)["verdict"] == "not_run"
    e = _e(good)
    e["rows"] = [r for r in e["rows"] if r["root"] != ROOTS[-1]]
    assert k0_sign(e, SPEC)["verdict"] == "not_run"
