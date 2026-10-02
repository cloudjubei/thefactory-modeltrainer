"""§C.49 D1 — the PRE-REGISTERED diagnosis of why the Connect-4 stop signal never fired (h88): on the six saved T10
nets, is the search or the net the blocker where they disagree, and are the nets certified through 10 plies anyway,
judged by harness.floor_plateau.plateau_report. Each claim's proof passes only on SUPPORTED; its undecidable proof
passes on INCONCLUSIVE or NOT_RUN. The register pins this file and harness/floor_plateau.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_plateau import plateau_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_D1_plateau.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return plateau_report(load_evidence(EVIDENCE / FILE))


def test_c49_D1_at_the_plateau_the_search_not_the_net_blocks_the_stop_signal():
    assert _report()["blocker"]["verdict"] == "supported"


def test_c49_D1_the_blocker_is_undecidable():
    assert _report()["blocker"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_D1_most_saved_nets_are_certified_through_10_plies_despite_never_stopping():
    assert _report()["certified"]["verdict"] == "supported"


def test_c49_D1_the_certified_count_is_undecidable():
    assert _report()["certified"]["verdict"] in {"inconclusive", "not_run"}
