"""Direct tests for harness/label_diagnosis.py — D4: at the nets' POLICY-SIDE errors (the raw move is wrong, yet the
value head rates an optimal move above it), does the net's own relabel search LABEL an optimal move? Broken out by
whether the position is on the net's own first-player tree (trained every iteration) and by ply."""
from __future__ import annotations

import pytest

from harness.label_diagnosis import D4_SPEC, d4_report

SPEC = {"seeds": (1, 2), "measurement_fp": "m" * 12, "config": {"plies": [0, 2], "sims": 4}, "min_errors": 4,
        "support_at": 0.6, "refute_at": 0.4}
VALUES = {0: 0, 1: 1}
POLICY_SIDE = {0: 0.2, 1: 0.5}
VALUE_SIDE = {0: 0.5, 1: 0.2}


def _row(seed=1, ply=0, label=1, q_hat=POLICY_SIDE, on_tree=False, raw=0, values=VALUES):
    return {"seed": seed, "ply": ply, "raw": raw, "label": label, "on_tree": on_tree,
            "q_hat": {str(a): q for a, q in q_hat.items()}, "values": {str(a): v for a, v in values.items()}}


def _evidence(rows):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "config": SPEC["config"], "seeds": list(SPEC["seeds"]),
            "positions": rows}


@pytest.mark.parametrize("n,right,verdict", [(5, 5, "supported"), (5, 3, "supported"), (4, 2, "inconclusive"),
                                             (5, 2, "refuted"), (5, 0, "refuted"), (3, 3, "inconclusive")])
def test_the_verdict_is_the_share_of_policy_side_errors_whose_label_is_optimal(n, right, verdict):
    rows = [_row(label=1)] * right + [_row(label=0)] * (n - right)
    r = d4_report(_evidence(rows), SPEC)
    assert r["integrity"] == [] and r["verdict"] == verdict and (r["policy_side"], r["label_optimal"]) == (n, right)


def test_value_side_errors_and_correct_moves_are_counted_apart_and_do_not_judge():
    rows = ([_row(label=1)] * 2 + [_row(q_hat=VALUE_SIDE, label=1)] * 3 + [_row(q_hat=VALUE_SIDE, label=0)]
            + [_row(raw=1, label=0)] * 4)
    r = d4_report(_evidence(rows), SPEC)
    assert (r["policy_side"], r["label_optimal"], r["verdict"]) == (2, 2, "inconclusive")
    assert r["value_side"] == {"errors": 4, "label_optimal": 3}


def test_a_label_counts_when_it_is_any_optimal_move():
    rows = [_row(label=2, q_hat={0: 0.1, 1: 0.5, 2: 0.3}, values={0: 0, 1: 1, 2: 1})] * 4
    assert d4_report(_evidence(rows), SPEC)["label_optimal"] == 4


def test_the_reading_is_broken_out_by_the_net_s_own_tree_and_by_ply():
    rows = [_row(on_tree=True, label=1), _row(on_tree=True, label=0), _row(ply=2, label=1), _row(seed=2, label=1)]
    r = d4_report(_evidence(rows), SPEC)
    assert r["by_tree"] == {"on_tree": {"policy_side": 2, "label_optimal": 1},
                            "off_tree": {"policy_side": 2, "label_optimal": 2}}
    assert r["by_ply"]["2"] == {"policy_side": 1, "label_optimal": 1}
    assert r["by_seed"]["2"] == {"policy_side": 1, "label_optimal": 1}


@pytest.mark.parametrize("breakage", ["fingerprint", "config", "seeds", "ply", "seed", "label"])
def test_evidence_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _evidence([_row()] * 4)
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "config":
        e["config"] = {"plies": [0, 2], "sims": 8}
    elif breakage == "seeds":
        e["seeds"] = [1]
    elif breakage == "ply":
        e["positions"] = [_row(ply=4)] + [_row()] * 3
    elif breakage == "seed":
        e["positions"] = [_row(seed=9)] + [_row()] * 3
    else:
        e["positions"] = [_row(label=5)] + [_row()] * 3
    r = d4_report(e, SPEC)
    assert r["verdict"] == "not_run" and r["integrity"]


def test_the_registered_measurement_is_d3_s_population_with_t10_s_relabel_search():
    from harness.value_diagnosis import D3_SPEC

    assert D4_SPEC["seeds"] == D3_SPEC["seeds"]
    assert D4_SPEC["config"] == {"plies": D3_SPEC["config"]["plies"], "sims": 200}
    assert (D4_SPEC["min_errors"], D4_SPEC["support_at"], D4_SPEC["refute_at"]) == (100, 0.6, 0.4)
