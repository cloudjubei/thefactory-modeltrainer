"""§C.42 item 1 — the PRE-REGISTERED localization claims (h11-h14), each judged by harness.ceiling from the stored
multi-seed evidence. The decision rule lives in harness.ceiling and was fixed, with these tests, before the run
that produced evidence/tictactoe_ceiling.json.gz; a test here fails exactly when its registered claim is false.

h12 and h14 were REFUTED by that run. Their tests now pin the refutation (h15, h16) — the register keeps the
original FAIL verdicts, drawn after pre-registration, and points from each refuted claim to its successor."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.ceiling import localize_report
from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence" / "tictactoe_ceiling.json.gz"
FILES = ("tictactoe_ceiling.json.gz", "tictactoe_base.json.gz", "tictactoe_mixed.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE.parent / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _report():
    return localize_report(load_evidence(EVIDENCE))


def test_h11_the_tictactoe_blind_spot_is_SYSTEMATIC_across_seeds():
    r = _report()
    assert r["n_seeds"] >= 7
    assert r["systematic"]["verdict"] is True


def test_h12_REFUTED_and_h15_the_deep_search_check_is_CONFOUNDED_by_search_alone():
    nb = _report()["net_not_budget"]
    assert nb["verdict"] is False and nb["budget_fixes"] is True
    assert nb["confounded"] is True and nb["attributable"] is False


def test_h13_failing_states_were_STARVED_of_training_visits():
    assert _report()["starved"]["verdict"] is True


def test_h14_REFUTED_and_h16_failing_states_are_NOT_enriched_off_the_optimal_play_manifold():
    om = _report()["off_manifold"]
    assert om["verdict"] is False and om["p"] > 0.5
    assert om["failing_off"] / om["failing_distinct"] <= om["base_rate"]


def test_h17_search_alone_does_most_of_the_work_and_learning_adds_the_rest():
    ev = load_evidence(EVIDENCE)
    ctrl = ev["search_alone"]
    trained = [s["coverage"] for s in ev["seeds"]]
    policy = [s["policy_coverage"] for s in ev["seeds"]]
    assert 0.90 < ctrl["coverage_eval"] < min(trained)
    assert ctrl["policy_coverage"] < 0.6 < 0.9 < min(policy)


BASE = EVIDENCE.parent / "tictactoe_base.json.gz"
MIXED = EVIDENCE.parent / "tictactoe_mixed.json.gz"


def _ab():
    from harness.ceiling import ab_report
    return ab_report(load_evidence(BASE), load_evidence(MIXED), load_evidence(EVIDENCE))


def test_h18_fresh_baseline_seeds_miss_the_localized_target_REPLICATION():
    r = _ab()
    assert len(r["seeds"]) >= 10
    assert r["replication"]["verdict"] is True


def test_h19_REFUTED_mixed_openings_do_NOT_raise_target_state_correctness():
    assert _ab()["target"]["verdict"] is False


def test_h20_REFUTED_mixed_openings_do_NOT_raise_raw_policy_coverage_at_alpha_over_four():
    assert _ab()["policy"]["verdict"] is False


def test_h21_REFUTED_mixed_openings_do_NOT_raise_eval_budget_coverage():
    assert _ab()["coverage"]["verdict"] is False


def test_h22_raising_self_play_visits_did_NOT_fix_the_target_so_starvation_is_not_the_cause():
    r = _ab()
    assert r["manipulation"]["verdict"] is True          # the treatment DID reach the target states
    assert r["target"]["verdict"] is False               # yet target correctness did not improve
    assert r["mechanism_causal"]["mechanism"] == "visits" and r["mechanism_causal"]["is_causal"] is False


def test_h23_exposure_raised_visits_but_repairs_at_most_a_third_of_the_blind_spot_and_not_the_prior():
    r = _ab()
    assert r["manipulation"]["verdict"] is True
    assert r["target"]["upper"] < 0.12
    assert r["target_policy"]["upper"] < 0.05
