"""§C.49 T13 — the PRE-REGISTERED claim that value targets for the strategy-tree positions do no harm on
tic-tac-toe: T9's process with the option on still plays perfectly from the start on at least 8 of 10 seeds, judged
by harness.floor_tree.tree_report on harness.floor_tree_value.SPEC. The proof passes only on SUPPORTED; the
undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file, harness/floor_tree_value.py,
harness/floor_tree.py and harness/floor_stop.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_tree import tree_report
from harness.floor_tree_value import SPEC, T13_ARM

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = f"c49_T13_{T13_ARM}.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return tree_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_T13_with_tree_value_targets_the_settled_net_still_plays_perfectly_from_the_start_on_8_of_10_seeds():
    assert _report()["pstart"]["verdict"] == "supported"


def test_c49_T13_is_undecidable():
    assert _report()["pstart"]["verdict"] in {"inconclusive", "not_run"}
