"""§C.49 T10 pilots — the two timed one-seed Connect-4 pilots that decided the solver-free process's shape,
registered AFTER the data as descriptions: with the 2-deep siblings an iteration is dominated by relabelling the
siblings; with the strategy tree instead of siblings and relabelling spread over 8 workers it is ~two orders of
magnitude faster."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
SIBLINGS = "c49_T10_pilot_siblings.json.gz"
TREE = "c49_T10_pilot_tree_parallel.json.gz"
FILES = (SIBLINGS, TREE)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_with_2_deep_siblings_a_connect4_iteration_is_over_90_percent_sibling_relabelling():
    e = load_evidence(EVIDENCE / SIBLINGS)
    assert e["config"]["reanalyze_siblings"] and e["config"]["sibling_depth"] == 2
    second = e["history"][1]
    seconds = e["pass_seconds"][1] - e["pass_seconds"][0]
    assert second["siblings"] > 50000 and seconds > 3600 and second["sibling_relabel_s"] / seconds > 0.9


def test_c49_with_the_tree_instead_of_siblings_and_8_relabel_workers_an_iteration_takes_under_a_minute():
    slow = load_evidence(EVIDENCE / SIBLINGS)
    e = load_evidence(EVIDENCE / TREE)
    assert not e["config"]["reanalyze_siblings"] and e["config"]["relabel_workers"] == 8
    passes = [p["seconds"] for p in e["passes"]]
    second = passes[1] - passes[0]
    assert second < 60 and (slow["pass_seconds"][1] - slow["pass_seconds"][0]) / second > 100
    assert all((h.get("siblings") or 0) == 0 for h in e["history"])
