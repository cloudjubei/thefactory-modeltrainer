"""§C.48 T1b — descriptive findings pinned to their evidence (registered after the data; not predictions).

T1 asked for the smallest NET that plays tic-tac-toe perfectly at every raw position from exact labels. T1b asks
for the smallest SETUP: the same supervised fit, with the input first mapped to its canonical image under the
verified isometries (the move mapped back), and/or given two rule features per move (wins now; hands the opponent a
win). Every setup is scored on all 4,520 raw positions — a canonicalising setup through its own map back.
Evidence: evidence/c48_T1b_setups.json.gz, with evidence/c48_T1_smallest.json.gz for the raw frontier."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c48_T1b_setups.json.gz", "c48_T1_smallest.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _setups():
    rows = load_evidence(EVIDENCE / FILES[0])["summary"] + load_evidence(EVIDENCE / FILES[1])["summary"]
    for r in rows:
        r.setdefault("input", "raw")
    return rows


def _frontier(rows, inputs):
    return min(r["params"] for r in rows if r["input"] in inputs and r["solved_seeds"] == r["seeds"] == 5)


def test_c48_T1b_every_setup_was_scored_on_every_raw_position():
    runs = load_evidence(EVIDENCE / FILES[0])["runs"]
    assert runs and all(r["positions"] == 4520 for r in runs)
    assert {r["train_rows"] for r in runs if r["spec"]["input"].startswith("canon")} == {627}


def test_c48_T1b_canonicalising_the_input_cuts_the_smallest_perfect_net_about_six_fold_from_5_6K_to_938():
    rows = _setups()
    raw, canon = _frontier(rows, {"raw"}), _frontier(rows, {"canon"})
    assert raw == 5629 and canon == 938 and raw / canon > 5.5
    assert all(r["solved_seeds"] < 5 for r in rows if r["input"] == "canon" and r["params"] < canon)


def test_c48_T1b_rule_features_do_not_lower_the_frontier_no_featured_setup_is_perfect_on_every_seed():
    rows = [r for r in _setups() if r["input"] in {"feat", "canon_feat"}]
    assert rows and all(r["solved_seeds"] < 5 for r in rows)
