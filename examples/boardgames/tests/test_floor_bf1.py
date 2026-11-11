"""Direct tests for harness/floor_bf1.py — the judge of §3.7's registered run: best-first proving with maps against
depth-first proving with maps and best-first proving with no maps, at the same work, on positions none of the three
cut short; and every proof sound."""
from __future__ import annotations

from harness.floor_bf1 import ARMS, SPEC, bf1_report, settings, verdict
from harness.floor_t8 import SPEC as T8

SOURCE = {"sha256": "t8"}
JUDGED = {**SPEC, "min_positions": 4}


def _row(best, depth, nomaps, value=1, ceiling=()):
    out = {"value": value, "arms": {}}
    for arm, result in zip(ARMS, (best, depth, nomaps)):
        out["arms"][arm] = {"result": result, "ceiling": arm in ceiling}
    return out


def _evidence(rows, spec=JUDGED, sha="t8"):
    return {"spec": "harness.floor_bf1", "settings": settings(spec), "source_sha256": sha, "positions": rows}


def _report(rows, check=lambda row, arm: True, spec=JUDGED, **kwargs):
    return bf1_report(_evidence(rows, spec, **kwargs), SOURCE, spec, check)


def test_the_spec_reads_t8_s_game_under_its_rule_with_the_pilots_settings():
    assert SPEC["source"] == T8["file"] == "c49_t8.json.gz" and SPEC["rule"] == T8["rule"]
    assert SPEC["file"] == "c49_bf1.json.gz" and SPEC["stones"] == (8, 10, 12, 14, 16, 18, 20, 22)
    assert (SPEC["per"], SPEC["evaluations"], SPEC["trials"], SPEC["bound"], SPEC["reuse"]) == (8, 4000, 50, 5000, 8)
    assert (SPEC["support"], SPEC["refute"], SPEC["min_positions"], SPEC["ceiling"]) == (2.0, 1.2, 60, 3600)
    assert ARMS == ("best", "depth", "nomaps")


def test_a_ratio_is_supported_at_twice_the_control_and_refuted_at_six_fifths():
    assert [verdict(t, 10, SPEC) for t in (20, 19, 13, 12, 5)] == [
        "supported", "inconclusive", "inconclusive", "refuted", "refuted"]
    assert verdict(1, 0, SPEC) == "supported" and verdict(0, 0, SPEC) == "refuted"


def test_the_counts_are_over_the_positions_no_arm_cut_short():
    rows = [_row(True, False, False)] * 4 + [_row(True, True, True, ceiling=("depth",)), _row(None, True, True)]
    r = _report(rows)
    assert r["included"] == 5 and r["proved"] == {"best": 4, "depth": 1, "nomaps": 1}
    assert r["best_vs_depth"] == "supported" and r["best_vs_nomaps"] == "supported"


def test_a_small_gain_refutes_and_a_middle_one_is_inconclusive():
    rows = [_row(True, True, True)] * 5 + [_row(True, False, True)] * 2
    r = _report(rows)
    assert r["proved"] == {"best": 7, "depth": 5, "nomaps": 7}
    assert r["best_vs_depth"] == "inconclusive" and r["best_vs_nomaps"] == "refuted"


def test_too_few_positions_is_inconclusive():
    r = _report([_row(True, False, False)] * 3)
    assert r["best_vs_depth"] == r["best_vs_nomaps"] == "inconclusive" and r["included"] == 3
    exactly = _report([_row(True, False, False)] * 4)
    assert exactly["included"] == 4 and exactly["best_vs_depth"] == exactly["best_vs_nomaps"] == "supported"


def test_every_proof_must_check_and_be_a_win_by_the_grader():
    rows = [_row(True, True, False)] * 4
    assert _report(rows)["sound"] == "supported"
    assert _report(rows, check=lambda row, arm: arm != "depth")["sound"] == "refuted"
    assert _report([_row(True, False, False, value=0)] + rows)["sound"] == "refuted"
    assert _report([_row(True, False, False, ceiling=("best",))] + rows,
                   check=lambda row, arm: not row["arms"]["best"]["ceiling"])["sound"] == "refuted"


def test_soundness_with_no_proof_at_all_is_inconclusive():
    assert _report([_row(None, False, None)] * 4)["sound"] == "inconclusive"


def test_a_run_off_its_spec_or_its_source_is_not_run():
    rows = [_row(True, False, False)] * 4
    neither = {"best_vs_depth": "not_run", "best_vs_nomaps": "not_run", "sound": "not_run"}
    assert {k: v for k, v in bf1_report({**_evidence(rows), "spec": "harness.floor_t8"}, SOURCE, JUDGED,
                                        lambda r, a: True).items() if k in neither} == neither
    assert {k: v for k, v in _report(rows, sha="other").items() if k in neither} == neither
    other = {**JUDGED, "evaluations": 400}
    assert {k: v for k, v in bf1_report(_evidence(rows, other), SOURCE, JUDGED, lambda r, a: True).items()
            if k in neither} == neither
