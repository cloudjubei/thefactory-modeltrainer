"""§C.11 ANALYSIS LEDGER — comparisons are DRAWN from a ledger, never hand-rolled.

Each test replays a mistake I made ON 2026-09-04, *after* the §C.9 guards existed — because the guards were
available but nothing forced measurements through them:
  L1 MIXED PROVENANCE: compared a gate-selected checkpoint against final ones inside one grid.
  L2 UNVERIFIED BUDGET: labelled a comparison "matched budget" when the arms had 9.6k vs 16k games.
  L3 UNCOUNTED MULTIPLICITY: ran ~6 paired tests on overlapping roots, then read p=0.039 as significant.
"""
import pytest

from harness.ledger import Ledger


def _mk(tmp_path):
    return Ledger(tmp_path / "ledger.json")


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


def test_refuses_comparison_across_different_root_families(tmp_path):
    # Pairing is only meaningful on IDENTICAL roots; different root_ids are not paired data.
    led = _mk(tmp_path)
    led.record("x", outcomes=[1] * 10, params=1, games=1, provenance="final", seed=99, roots_id="R1")
    led.record("y", outcomes=[1] * 10, params=1, games=1, provenance="final", seed=7, roots_id="R2")
    with pytest.raises(ValueError, match="(?i)root"):
        led.compare("x", "y")


def test_ledger_persists_across_instances(tmp_path):
    led = _mk(tmp_path)
    led.record("p", outcomes=[1, 0, 1], params=1, games=1, provenance="final", seed=99, roots_id="R")
    led.compare("p", "p")
    again = Ledger(tmp_path / "ledger.json")
    assert "p" in again.entries()
    assert again.compare("p", "p")["comparisons_on_family"] == 2  # multiplicity survives a restart
