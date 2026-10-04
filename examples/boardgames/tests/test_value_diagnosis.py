"""Direct tests for harness/value_diagnosis.py — D3: does a net's VALUE HEAD back its own wrong moves? At a position
with exact move values, the net's raw move is an ERROR when it is not optimal; the value head BACKS it when it rates
the raw move's resulting position at least as high (mover's view) as every optimal move's."""
from __future__ import annotations

import pytest

from harness.value_diagnosis import d3_report, position_reading

SPEC = {"seeds": (1, 2), "measurement_fp": "m" * 12, "config": {"plies": [0, 2]}, "min_errors": 4,
        "support_at": 0.6, "refute_at": 0.4}


@pytest.mark.parametrize("raw,q_hat,values,reading", [
    (0, {0: 0.5, 1: 0.2}, {0: 0, 1: 1}, {"error": True, "backs_raw": True, "misranks": True}),
    (0, {0: 0.2, 1: 0.5}, {0: 0, 1: 1}, {"error": True, "backs_raw": False, "misranks": False}),
    (0, {0: 0.5, 1: 0.5}, {0: 0, 1: 1}, {"error": True, "backs_raw": True, "misranks": True}),
    (1, {0: 0.5, 1: 0.2, 2: 0.1}, {0: -1, 1: 1, 2: 1}, {"error": False, "backs_raw": False, "misranks": True}),
    (1, {0: 0.1, 1: 0.2, 2: 0.6}, {0: -1, 1: 1, 2: 1}, {"error": False, "backs_raw": False, "misranks": False}),
    (2, {0: 0.9, 1: 0.3, 2: 0.2}, {0: 0, 1: 1, 2: -1}, {"error": True, "backs_raw": False, "misranks": True}),
    (0, {0: 0.5, 1: 0.9, 2: 0.2}, {0: 0, 1: 1, 2: 1}, {"error": True, "backs_raw": False, "misranks": False}),
])
def test_a_position_reads_whether_the_raw_move_errs_and_whether_the_value_head_backs_it(raw, q_hat, values, reading):
    assert position_reading(raw, q_hat, values) == reading


def test_a_position_where_every_move_is_optimal_has_no_reading():
    assert position_reading(0, {0: 0.1, 1: 0.9}, {0: 1, 1: 1}) is None


def test_the_value_head_must_rate_exactly_the_legal_moves():
    with pytest.raises(ValueError, match="same moves"):
        position_reading(0, {0: 0.1}, {0: 1, 1: 0})


def _row(seed, ply, raw, q_hat, values):
    return {"seed": seed, "ply": ply, "raw": raw, "q_hat": {str(a): q for a, q in q_hat.items()},
            "values": {str(a): v for a, v in values.items()}}


BACKED = ({0: 0.5, 1: 0.2}, {0: 0, 1: 1})
UNBACKED = ({0: 0.2, 1: 0.5}, {0: 0, 1: 1})
RIGHT = ({0: 0.1, 1: 0.5}, {0: 0, 1: 1})


def _evidence(per_seed):
    rows = [_row(seed, ply, 0 if kind is not RIGHT else 1, *kind) for seed, kinds in per_seed.items()
            for ply, kind in kinds]
    return {"measurement_fingerprint": SPEC["measurement_fp"], "config": SPEC["config"], "seeds": list(SPEC["seeds"]),
            "positions": rows}


@pytest.mark.parametrize("backed,verdict", [(4, "supported"), (3, "supported"), (2, "inconclusive"),
                                            (1, "refuted"), (0, "refuted")])
def test_the_verdict_is_the_pooled_share_of_errors_the_value_head_backs(backed, verdict):
    kinds = [(0, BACKED)] * backed + [(2, UNBACKED)] * (4 - backed)
    r = d3_report(_evidence({1: kinds[:2] + [(0, RIGHT)], 2: kinds[2:]}), SPEC)
    assert r["integrity"] == [] and r["verdict"] == verdict and r["errors"] == 4 and r["backed"] == backed


@pytest.mark.parametrize("backed,verdict", [(3, "supported"), (2, "refuted")])
def test_a_share_exactly_on_a_bar_takes_that_bar_s_verdict(backed, verdict):
    kinds = [(0, BACKED)] * backed + [(2, UNBACKED)] * (5 - backed)
    r = d3_report(_evidence({1: kinds[:3], 2: kinds[3:]}), SPEC)
    assert r["errors"] == 5 and r["verdict"] == verdict


def test_too_few_errors_is_inconclusive_and_the_reading_is_broken_out_by_net_and_ply():
    r = d3_report(_evidence({1: [(0, BACKED), (2, UNBACKED)], 2: [(0, BACKED), (2, RIGHT)]}), SPEC)
    assert r["verdict"] == "inconclusive" and r["errors"] == 3
    assert r["by_seed"]["1"] == {"positions": 2, "errors": 2, "backed": 1, "misranked_when_right": 0, "right": 0}
    assert r["by_ply"]["2"] == {"positions": 2, "errors": 1, "backed": 0, "misranked_when_right": 0, "right": 1}


def test_the_control_counts_correct_positions_where_the_value_head_still_misranks():
    misranked_right = ({0: 0.9, 1: 0.5}, {0: 0, 1: 1})
    rows = [_row(1, 0, 1, *misranked_right), _row(1, 0, 1, *RIGHT)]
    e = {**_evidence({}), "positions": rows}
    assert d3_report(e, SPEC)["by_seed"]["1"]["misranked_when_right"] == 1


@pytest.mark.parametrize("breakage", ["fingerprint", "config", "seeds", "ply", "unknown seed"])
def test_evidence_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _evidence({1: [(0, BACKED)] * 4, 2: []})
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "config":
        e["config"] = {"plies": [0]}
    elif breakage == "seeds":
        e["seeds"] = [1]
    elif breakage == "ply":
        e["positions"][0]["ply"] = 4
    else:
        e["positions"][0]["seed"] = 9
    r = d3_report(e, SPEC)
    assert r["verdict"] == "not_run" and r["integrity"]


def test_the_registered_measurement_reads_the_six_t10_nets_at_white_s_plies_through_8():
    from harness.value_diagnosis import D3_SPEC

    assert D3_SPEC["seeds"] == (361, 362, 363, 364, 365, 366) and D3_SPEC["config"] == {"plies": [0, 2, 4, 6, 8]}
    assert (D3_SPEC["min_errors"], D3_SPEC["support_at"], D3_SPEC["refute_at"]) == (100, 0.6, 0.4)
