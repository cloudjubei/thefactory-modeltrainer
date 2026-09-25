"""§C.47 leg C, Stage 0 — the PRE-REGISTERED Connect-4 go/no-go, judged by harness.stage0.stage0_report from
evidence/c47_C0.json.gz (five C_R nets, seeds 201-205, trained with the self-play recorder on and the solver
forbidden). The proof passes only on GO; the undecidable proof passes on UNDECIDED or NOT_RUN; STOP reads as the
claim refuted. The register pins this file and harness/stage0.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.stage0 import stage0_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c47_C0.json.gz",)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return stage0_report(load_evidence(EVIDENCE / FILES[0]))


def test_c47_C0_the_direct_sibling_channel_has_headroom_on_connect4():
    assert _report()["C0"]["verdict"] == "go"


def test_c47_C0_is_undecidable():
    assert _report()["C0"]["verdict"] in {"undecided", "not_run"}
