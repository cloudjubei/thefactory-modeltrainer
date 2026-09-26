"""§C.48 T1 and T3 — descriptive findings pinned to their evidence (registered after the data; not predictions).

T1 (evidence/c48_T1_smallest.json.gz): the smallest net in the harness's family that plays tic-tac-toe perfectly at
every one of the 4,520 raw positions, learned from exact labels with no symmetry, trained to convergence.
T3 (evidence/c48_T3_ceiling.json.gz): held-out accuracy on non-trivial Connect-4 positions by net size and number of
exactly labelled training positions, from the pre-solved bank (evidence/c48_bank.json.gz)."""
from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c48_T1_smallest.json.gz", "c48_T3_ceiling.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _t1():
    return {s["name"]: s for s in load_evidence(EVIDENCE / FILES[0])["summary"]}


@lru_cache(maxsize=1)
def _t3():
    cells: dict = {}
    for r in load_evidence(EVIDENCE / FILES[1])["runs"]:
        cells.setdefault((r["net"], r["train_size"]), []).append(r["readings"])
    return {k: {m: statistics.mean(x[m] for x in v) for m in v[0]} for k, v in cells.items()}


def test_c48_T1_about_5K_parameters_hold_perfect_tic_tac_toe_and_under_2K_never_do():
    t1 = _t1()
    assert t1["residual16"]["params"] == 5629 and t1["residual16"]["solved_seeds"] == 5
    assert t1["legacy32"]["solved_seeds"] == 5 and t1["legacy16"]["solved_seeds"] == 3
    assert all(s["solved_seeds"] == 0 for s in t1.values() if s["params"] <= 1866)


def test_c48_T3_connect4_accuracy_is_DATA_limited_every_net_improves_with_every_tripling_of_exact_labels():
    t3 = _t3()
    for net in ("legacy32", "residual32", "residual64", "residual128"):
        sizes = sorted((n for (m, n) in t3 if m == net))
        accs = [t3[(net, n)]["test_nontrivial"] for n in sizes]
        assert accs == sorted(accs) and accs[-1] - accs[0] > 0.1


def test_c48_T3_the_largest_net_reaches_about_82_percent_on_unseen_non_trivial_positions_and_fits_its_training_set():
    big = max((k for k in _t3() if k[0] == "residual128"), key=lambda k: k[1])
    r = _t3()[big]
    assert 0.80 <= r["test_nontrivial"] <= 0.84 and r["train_fit"] >= 0.97


def test_c48_T3_the_small_nets_cannot_even_fit_their_training_positions():
    t3 = _t3()
    for net in ("legacy32", "residual32"):
        full = max(n for (m, n) in t3 if m == net)
        assert t3[(net, full)]["train_fit"] < 0.93
