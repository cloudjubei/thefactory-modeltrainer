"""§C.49 T9 — the PRE-REGISTERED claims that the strategy tree can replace the siblings: with siblings off, T8's
process still ends with a net that plays perfectly from the start as the first player, and its stop signal stays
safe and timely, judged by harness.floor_tree.tree_report. Each claim's proof passes only on SUPPORTED; its
undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file, harness/floor_tree.py and
harness/floor_stop.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_tree import T9_ARM, tree_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = f"c49_T9_{T9_ARM}.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return tree_report(load_evidence(EVIDENCE / FILE))


def test_c49_T9_without_siblings_the_settled_net_plays_perfectly_from_the_start_on_8_of_10_seeds():
    assert _report()["pstart"]["verdict"] == "supported"


def test_c49_T9_the_start_reading_is_undecidable():
    assert _report()["pstart"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_T9_without_siblings_the_stop_signal_never_stops_early_and_stops_in_time_on_8_of_10_seeds():
    assert _report()["stop"]["verdict"] == "supported"


def test_c49_T9_the_stop_reading_is_undecidable():
    assert _report()["stop"]["verdict"] in {"inconclusive", "not_run"}
