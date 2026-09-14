"""§C.11 ANALYSIS LEDGER — comparisons are DRAWN from a ledger, never hand-rolled.

Each test replays a mistake I made ON 2026-09-04, *after* the §C.9 guards existed — because the guards were
available but nothing forced measurements through them:
  L1 MIXED PROVENANCE: compared a gate-selected checkpoint against final ones inside one grid.
  L2 UNVERIFIED BUDGET: labelled a comparison "matched budget" when the arms had 9.6k vs 16k games.
  L3 UNCOUNTED MULTIPLICITY: ran ~6 paired tests on overlapping roots, then read p=0.039 as significant.
"""
import json

import pytest

from harness.ledger import Ledger, run_budget


def _mk(tmp_path):
    return Ledger(tmp_path / "ledger.json")


def _mk_run(tmp_path, name, rows, cfg=None):
    d = tmp_path / name
    d.mkdir()
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        (d / f"ckpt_{r['batch']}.pt").write_text("x")
    if cfg is not None:
        (tmp_path / f"{name}.json").write_text(json.dumps(cfg))
    return d


def test_records_carry_full_provenance(tmp_path):
    led = _mk(tmp_path)
    led.record("big_9k", outcomes=[1] * 109 + [0] * 19, params=1_785_873, games=9600,
               provenance="final", seed=99, roots_id="e24_s99_n128")
    e = led.entries()["big_9k"]
    assert e["n"] == 128 and e["converted"] == 109 and e["provenance"] == "final"
    assert e["games"] == 9600 and e["roots_id"] == "e24_s99_n128"
    assert 0.77 < e["ci"][0] < e["rate"] < e["ci"][1] < 0.91   # CI always attached, never a bare rate


def test_L1_refuses_mixed_provenance(tmp_path):
    # The exact error: carry's GATE-SELECTED champion put in a grid beside FINAL checkpoints.
    led = _mk(tmp_path)
    led.record("small_final", outcomes=[1] * 107 + [0] * 21, params=302_353, games=16000,
               provenance="final", seed=99, roots_id="R")
    led.record("small_gated", outcomes=[1] * 99 + [0] * 29, params=302_353, games=9600,
               provenance="gate_selected", seed=99, roots_id="R")
    with pytest.raises(ValueError, match="(?i)provenance"):
        led.compare("small_final", "small_gated")
    ok = led.compare("small_final", "small_gated", allow_mixed_provenance=True)
    assert ok["provenance_warning"] and "gate_selected" in ok["provenance_warning"]


def test_L2_flags_unmatched_budgets(tmp_path):
    led = _mk(tmp_path)
    led.record("a", outcomes=[1] * 109 + [0] * 19, params=1_785_873, games=9600,
               provenance="final", seed=99, roots_id="R")
    led.record("b", outcomes=[1] * 107 + [0] * 21, params=302_353, games=16000,
               provenance="final", seed=99, roots_id="R")
    r = led.compare("a", "b")
    assert r["budget_matched"] is False
    assert "9600" in r["budget_note"] and "16000" in r["budget_note"]
    r2 = led.compare("a", "a")
    assert r2["budget_matched"] is True


def test_L3_counts_multiplicity_on_the_same_roots(tmp_path):
    # Six paired tests on one root family: the corrected threshold must move, and a p=0.039 that looked
    # significant must stop being reported as such.
    led = _mk(tmp_path)
    for i in range(4):
        led.record(f"n{i}", outcomes=([1] * (100 + i) + [0] * (28 - i)), params=1000 * (i + 1),
                   games=9600, provenance="final", seed=99, roots_id="R")
    for a, b in [("n0", "n1"), ("n0", "n2"), ("n0", "n3"), ("n1", "n2"), ("n1", "n3")]:
        led.compare(a, b)
    r = led.compare("n2", "n3")
    assert r["comparisons_on_family"] == 6
    assert r["alpha_corrected"] == pytest.approx(0.05 / 6, rel=1e-6)
    verdict = led.significant(p=0.039, roots_id="R")
    assert verdict is False, "p=0.039 must NOT count as significant after 6 comparisons"
    assert led.significant(p=0.001, roots_id="R") is True


def test_budget_matched_checkpoint_compares_against_a_final_one(tmp_path):
    # 2026-09-08: the bins A/B ran 24 batches; its control run was configured for 40. The like-for-like
    # comparison is at the SAME batch index, so the control arm is ckpt_23 of 40 — not a "final" checkpoint,
    # but not score-selected either. Under label-equality that comparison raised, and the only way through was
    # allow_mixed_provenance=True, which also switches off the L1 guard. Selection status, not the label, is
    # what a comparison has to agree on.
    led = _mk(tmp_path)
    led.record("bins_final", outcomes=[1] * 340 + [0] * 44, params=302_353, games=9600,
               provenance="final", seed=257, roots_id="R")
    led.record("ctrl_b23", outcomes=[1] * 333 + [0] * 51, params=302_353, games=9600,
               provenance="budget_matched", seed=257, roots_id="R")
    r = led.compare("bins_final", "ctrl_b23")
    assert r["budget_matched"] is True
    assert "budget_matched" in r["provenance_warning"]  # visible, but not a refusal
    assert "p" in r


def test_budget_matched_with_unequal_budgets_is_a_contradiction(tmp_path):
    # "budget_matched" claims the checkpoint was pinned to the other arm's budget. If the budgets then differ,
    # the label is false and the comparison is the L2 error wearing a reassuring name.
    led = _mk(tmp_path)
    led.record("a", outcomes=[1] * 100 + [0] * 28, params=1, games=9600,
               provenance="final", seed=99, roots_id="R")
    led.record("b", outcomes=[1] * 99 + [0] * 29, params=1, games=16000,
               provenance="budget_matched", seed=99, roots_id="R")
    with pytest.raises(ValueError, match="(?i)budget"):
        led.compare("a", "b")


def test_arbitrary_is_treated_as_selected(tmp_path):
    # An unexplained checkpoint gets the pessimistic reading: we cannot show it was not picked by looking.
    led = _mk(tmp_path)
    led.record("f", outcomes=[1] * 100 + [0] * 28, params=1, games=9600,
               provenance="final", seed=99, roots_id="R")
    led.record("u", outcomes=[1] * 99 + [0] * 29, params=1, games=9600,
               provenance="arbitrary", seed=99, roots_id="R")
    with pytest.raises(ValueError, match="(?i)provenance"):
        led.compare("f", "u")


def test_refuses_comparison_across_different_root_families(tmp_path):
    # Pairing is only meaningful on IDENTICAL roots; different root_ids are not paired data.
    led = _mk(tmp_path)
    led.record("x", outcomes=[1] * 10, params=1, games=1, provenance="final", seed=99, roots_id="R1")
    led.record("y", outcomes=[1] * 10, params=1, games=1, provenance="final", seed=7, roots_id="R2")
    with pytest.raises(ValueError, match="(?i)root"):
        led.compare("x", "y")


def test_run_budget_reads_the_final_checkpoint_off_the_run(tmp_path):
    d = _mk_run(tmp_path, "r", [{"batch": i, "games": 400} for i in range(3)])
    assert run_budget(d / "ckpt_2.pt") == {"batch": 2, "games": 1200, "provenance": "final", "code": None,
                                           "run_complete": None, "sims": None, "simulations": None}


def test_run_budget_labels_an_earlier_index_budget_matched(tmp_path):
    # The bins A/B ran 24 batches against a control configured for 40: batch 23 of the control is the
    # like-for-like point, and it must carry the budget of 24 batches, not of 40.
    d = _mk_run(tmp_path, "r", [{"batch": i, "games": 400} for i in range(40)])
    assert run_budget(d / "ckpt_23.pt")["provenance"] == "budget_matched"
    assert run_budget(d / "ckpt_23.pt")["games"] == 9600


def test_run_budget_picks_up_the_training_fingerprint_the_run_recorded(tmp_path):
    d = _mk_run(tmp_path, "r", [{"batch": i, "games": 400} for i in range(3)])
    (d / "provenance.json").write_text(json.dumps({"training_fingerprint": "5a55087160db"}))
    assert run_budget(d / "ckpt_2.pt")["code"] == "5a55087160db"


def test_run_budget_derives_games_for_runs_that_predate_cost_accounting(tmp_path):
    # carry_03 finished before `games` was recorded per batch; its rows carry cumulative `iterations_done`,
    # and the run's config says how many games an iteration plays. Guessing is not an option here — an
    # unverifiable budget is exactly the L2 error, so the number has to come from the run's own artefacts.
    d = _mk_run(tmp_path, "old", [{"batch": i, "iterations_done": 5 * (i + 1)} for i in range(40)],
                cfg={"games": 80, "iters_per_batch": 5})
    assert run_budget(d / "ckpt_23.pt") == {"batch": 23, "games": 9600, "provenance": "budget_matched",
                                            "code": None, "run_complete": None, "sims": None,
                                            "simulations": None}


def test_run_budget_refuses_when_the_budget_cannot_be_established(tmp_path):
    d = _mk_run(tmp_path, "mystery", [{"batch": i, "final_loss": 1.0} for i in range(3)])
    with pytest.raises(ValueError, match="(?i)budget"):
        run_budget(d / "ckpt_2.pt")


def test_run_budget_refuses_a_checkpoint_the_run_does_not_account_for(tmp_path):
    d = _mk_run(tmp_path, "gappy", [{"batch": i, "games": 400} for i in (0, 1, 3)])
    with pytest.raises(ValueError, match="(?i)batch"):
        run_budget(d / "ckpt_3.pt")


def _declare(run_dir, batches: int) -> None:
    (run_dir / "provenance.json").write_text(json.dumps({"request": {"batches": batches}}))


def test_run_budget_will_not_call_a_checkpoint_final_before_the_run_reaches_its_declared_end(tmp_path):
    """L1 through a DERIVED label: 'final' was 'is the last row in metrics.jsonl', which on a LIVE run is a
    snapshot of when you looked, not a property of the checkpoint. Worse, a run stopped early at a
    nice-looking batch would launder that choice into the most trusted label in the system."""
    d = _mk_run(tmp_path, "live", [{"batch": i, "games": 400} for i in range(12)])
    _declare(d, 24)
    b = run_budget(d / "ckpt_11.pt")
    assert b["provenance"] == "budget_matched" and b["run_complete"] is False


def test_run_budget_calls_the_last_checkpoint_final_once_the_run_reached_its_declared_end(tmp_path):
    d = _mk_run(tmp_path, "done", [{"batch": i, "games": 400} for i in range(24)])
    _declare(d, 24)
    b = run_budget(d / "ckpt_23.pt")
    assert b["provenance"] == "final" and b["run_complete"] is True


def test_run_budget_still_labels_earlier_indices_budget_matched_in_a_completed_run(tmp_path):
    d = _mk_run(tmp_path, "done", [{"batch": i, "games": 400} for i in range(24)])
    _declare(d, 24)
    assert run_budget(d / "ckpt_11.pt")["provenance"] == "budget_matched"


def test_run_budget_cannot_certify_completeness_without_a_declared_end(tmp_path):
    d = _mk_run(tmp_path, "undeclared", [{"batch": i, "games": 400} for i in range(3)])
    (d / "provenance.json").write_text(json.dumps({"training_fingerprint": "5a55087160db"}))
    b = run_budget(d / "ckpt_2.pt")
    assert b["run_complete"] is None and b["provenance"] == "final"


def test_compare_warns_when_an_arm_came_from_a_run_that_never_reached_its_declared_end(tmp_path):
    led = _mk(tmp_path)
    for name, games in (("mid", 4800), ("early", 1600)):
        led.record(name, outcomes=[1] * 5 + [0] * 5, params=1, games=games, provenance="budget_matched",
                   seed=131, roots_id="R", code="C", config="G", run_complete=False)
    r = led.compare("mid", "early", treatment="budget")
    assert "did not reach its declared end" in r["completeness_warning"]
    assert "mid" in r["completeness_warning"] and "early" in r["completeness_warning"]


def test_compare_is_quiet_about_completeness_when_both_runs_finished(tmp_path):
    led = _mk(tmp_path)
    for name, games in (("late", 9600), ("early", 1600)):
        led.record(name, outcomes=[1] * 5 + [0] * 5, params=1, games=games, provenance="budget_matched",
                   seed=131, roots_id="R", code="C", config="G", run_complete=True)
    assert led.compare("late", "early", treatment="budget")["completeness_warning"] == ""


def test_run_budget_refuses_a_checkpoint_not_named_by_batch(tmp_path):
    d = _mk_run(tmp_path, "champ", [{"batch": 0, "games": 400}])
    (d / "champion.pt").write_text("x")
    with pytest.raises(ValueError, match="(?i)provenance"):
        run_budget(d / "champion.pt")


def test_refuses_arms_trained_by_different_code(tmp_path):
    # The 2026-09-08 incident: both the gpool and categorical-head A/Bs reused controls trained under an older
    # Adam regime, so each varied two things at once. Provenance, budget and roots all checked out; nothing knew
    # what CODE wrote the checkpoints.
    led = _mk(tmp_path)
    led.record("arm", outcomes=[1] * 337 + [0] * 47, params=1, games=9600, provenance="final",
               seed=257, roots_id="R", code="5a55087160db")
    led.record("old_ctrl", outcomes=[1] * 341 + [0] * 43, params=1, games=9600, provenance="final",
               seed=257, roots_id="R", code="bceb94d254eb")
    with pytest.raises(ValueError, match="(?i)training code|fingerprint"):
        led.compare("arm", "old_ctrl")


def test_same_code_compares_cleanly(tmp_path):
    led = _mk(tmp_path)
    for name in ("a", "b"):
        led.record(name, outcomes=[1] * 100 + [0] * 28, params=1, games=9600, provenance="final",
                   seed=257, roots_id="R", code="5a55087160db")
    assert led.compare("a", "b")["code_warning"] == ""


def test_unknown_code_warns_rather_than_passing_silently(tmp_path):
    # Every run predating the fingerprint has no record of its training code. Silence would read as agreement,
    # which is exactly the state the system was in while two experiments were invalid.
    led = _mk(tmp_path)
    led.record("known", outcomes=[1] * 100 + [0] * 28, params=1, games=9600, provenance="final",
               seed=257, roots_id="R", code="5a55087160db")
    led.record("legacy", outcomes=[1] * 99 + [0] * 29, params=1, games=9600, provenance="final",
               seed=257, roots_id="R")
    r = led.compare("known", "legacy")
    assert "legacy" in r["code_warning"] and "unknown" in r["code_warning"].lower()


def _pair(led, code_a, code_b, cfg_a="C", cfg_b="C"):
    led.record("a", outcomes=[1] * 100 + [0] * 28, params=1, games=9600, provenance="final",
               seed=257, roots_id="R", code=code_a, config=cfg_a)
    led.record("b", outcomes=[1] * 99 + [0] * 29, params=1, games=9600, provenance="final",
               seed=257, roots_id="R", code=code_b, config=cfg_b)
    return led


def test_code_can_be_the_treatment_when_the_config_is_held_fixed(tmp_path):
    # Measuring the §C.14 optimizer change IS a code experiment: ctrl302_postfix_s0 vs carry_03 differ in the
    # harness and in nothing else. Refusing it would push the measurement outside the ledger, which is the
    # bypass every guard here exists to prevent — so the caller declares WHICH dimension is under test.
    led = _pair(_mk(tmp_path), "5a55087160db", "bceb94d254eb")
    r = led.compare("a", "b", treatment="code")
    assert "p" in r and r["code_warning"] == ""


def test_code_as_treatment_still_requires_the_config_to_match(tmp_path):
    # Otherwise "treatment=code" is just an opt-out from the code check wearing an experimental name.
    led = _pair(_mk(tmp_path), "5a55087160db", "bceb94d254eb", cfg_a="C1", cfg_b="C2")
    with pytest.raises(ValueError, match="(?i)config"):
        led.compare("a", "b", treatment="code")


def test_code_as_treatment_refuses_when_the_code_is_actually_the_same(tmp_path):
    # Nothing is being measured: the label would misdescribe the experiment.
    led = _pair(_mk(tmp_path), "5a55087160db", "5a55087160db")
    with pytest.raises(ValueError, match="(?i)same training code"):
        led.compare("a", "b", treatment="code")


def test_config_treatment_is_the_default_and_still_refuses_mixed_code(tmp_path):
    led = _pair(_mk(tmp_path), "5a55087160db", "bceb94d254eb", cfg_a="C1", cfg_b="C2")
    with pytest.raises(ValueError, match="(?i)training code"):
        led.compare("a", "b")


def test_config_treatment_requires_the_configs_to_actually_differ(tmp_path):
    # Two arms with identical configs and identical code are the same experiment run twice, not an A/B.
    led = _pair(_mk(tmp_path), "5a55087160db", "5a55087160db")
    with pytest.raises(ValueError, match="(?i)same config"):
        led.compare("a", "b")


def test_unknown_config_does_not_block_the_default_comparison(tmp_path):
    # Legacy entries carry no config fingerprint; that must warn, not refuse, or every historical entry becomes
    # uncomparable and the ledger stops being used.
    led = _mk(tmp_path)
    led.record("a", outcomes=[1] * 100 + [0] * 28, params=1, games=9600, provenance="final",
               seed=257, roots_id="R", code="5a55087160db")
    led.record("b", outcomes=[1] * 99 + [0] * 29, params=1, games=9600, provenance="final",
               seed=257, roots_id="R", code="5a55087160db")
    assert "p" in led.compare("a", "b")


def test_ledger_persists_across_instances(tmp_path):
    led = _mk(tmp_path)
    led.record("p", outcomes=[1, 0, 1], params=1, games=1, provenance="final", seed=99, roots_id="R")
    led.compare("p", "p")
    again = Ledger(tmp_path / "ledger.json")
    assert "p" in again.entries()
    assert again.compare("p", "p")["comparisons_on_family"] == 2  # multiplicity survives a restart


def _curve(led, games_a, games_b, code_a="ac11124cf17a", code_b="ac11124cf17a", cfg_a="R", cfg_b="R"):
    led.record("late", outcomes=[1] * 200 + [0] * 56, params=1, games=games_a, provenance="final",
               seed=131, roots_id="X", code=code_a, config=cfg_a)
    led.record("early", outcomes=[1] * 150 + [0] * 106, params=1, games=games_b, provenance="budget_matched",
               seed=131, roots_id="X", code=code_b, config=cfg_b)
    return led


def test_budget_can_be_the_treatment_for_a_learning_curve(tmp_path):
    # "Is the process learning?" compares ckpt_23 with ckpt_3 of ONE run: same code, same recipe, and the training
    # budget IS the variable. Under the default treatment the L2 guard refuses this (budgets differ, and the
    # earlier index carries the budget_matched label) — which would push the question outside the ledger.
    led = _curve(_mk(tmp_path), 9600, 1600)
    r = led.compare("late", "early", treatment="budget")
    assert "p" in r and r["budget_matched"] is False


def test_budget_treatment_requires_the_budgets_to_differ(tmp_path):
    led = _curve(_mk(tmp_path), 9600, 9600)
    with pytest.raises(ValueError, match="(?i)same budget"):
        led.compare("late", "early", treatment="budget")


def test_budget_treatment_holds_code_and_config_fixed(tmp_path):
    led = _curve(_mk(tmp_path), 9600, 1600, code_b="bceb94d254eb")
    with pytest.raises(ValueError, match="(?i)training code"):
        led.compare("late", "early", treatment="budget")
    led2 = _curve(_mk(tmp_path / "b"), 9600, 1600, cfg_b="S")
    with pytest.raises(ValueError, match="(?i)config"):
        led2.compare("late", "early", treatment="budget")


def test_default_treatment_still_refuses_a_learning_curve_pair(tmp_path):
    # The guard that motivated the new treatment must keep firing when nobody declared budget as the variable.
    led = _curve(_mk(tmp_path), 9600, 1600)
    with pytest.raises(ValueError, match="(?i)budget"):
        led.compare("late", "early")


def _declare_sims(run_dir, batches: int, sims: int, name: str, tmp_path) -> None:
    (run_dir / "provenance.json").write_text(json.dumps({"request": {"batches": batches, "sims": sims}}))
    (tmp_path / f"{name}.json").write_text(json.dumps({"sims": sims}))


def test_run_budget_reports_the_search_budget_and_total_simulations(tmp_path):
    """§C.26: `games` is a proxy for cost. The real budget of a search-based learner is SIMULATIONS, and two
    arms at different sims are comparable only on that."""
    d = _mk_run(tmp_path, "r", [{"batch": i, "games": 400} for i in range(12)])
    _declare_sims(d, 48, 96, "r", tmp_path)
    b = run_budget(d / "ckpt_11.pt")
    assert b["sims"] == 96 and b["simulations"] == 4800 * 96


def test_run_budget_leaves_simulations_unknown_when_the_run_never_declared_sims(tmp_path):
    d = _mk_run(tmp_path, "r", [{"batch": i, "games": 400} for i in range(3)])
    b = run_budget(d / "ckpt_2.pt")
    assert b["sims"] is None and b["simulations"] is None


def test_compare_accepts_a_compute_matched_pair_whose_game_counts_differ_by_design(tmp_path):
    """The pre-registered efficiency read: 96 sims x N games vs 32 sims x 3N games is the SAME compute, and the
    ledger must not reject it as a mislabelled budget just because `games` differs."""
    led = _mk(tmp_path)
    led.record("a11", outcomes=[1] * 6 + [0] * 4, params=1, games=4800, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G96", compute=4800 * 96)
    led.record("b35", outcomes=[1] * 8 + [0] * 2, params=1, games=14400, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G32", compute=14400 * 32)
    r = led.compare("a11", "b35", treatment="config")
    assert r["compute_matched"] and not r["budget_matched"]
    assert "COMPUTE-MATCHED" in r["budget_note"] and "460800" in r["budget_note"]


def test_compare_still_refuses_a_pair_that_matches_on_neither_games_nor_compute(tmp_path):
    led = _mk(tmp_path)
    led.record("a", outcomes=[1] * 6 + [0] * 4, params=1, games=4800, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G96", compute=4800 * 96)
    led.record("b", outcomes=[1] * 8 + [0] * 2, params=1, games=9600, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G32", compute=9600 * 32)
    with pytest.raises(ValueError, match="(?i)label is false"):
        led.compare("a", "b", treatment="config")


def test_compute_matching_does_not_paper_over_unknown_compute(tmp_path):
    led = _mk(tmp_path)
    led.record("a", outcomes=[1] * 6 + [0] * 4, params=1, games=4800, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G96")
    led.record("b", outcomes=[1] * 8 + [0] * 2, params=1, games=14400, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G32")
    with pytest.raises(ValueError, match="(?i)label is false"):
        led.compare("a", "b", treatment="config")


def test_a_budget_treatment_learning_curve_is_unaffected_by_compute_matching(tmp_path):
    led = _mk(tmp_path)
    for name, games in (("late", 19200), ("early", 4800)):
        led.record(name, outcomes=[1] * 5 + [0] * 5, params=1, games=games, provenance="budget_matched",
                   seed=131, roots_id="R", code="C", config="G", compute=games * 96)
    r = led.compare("late", "early", treatment="budget")
    assert not r["compute_matched"] and "BUDGETS DIFFER" in r["budget_note"]
