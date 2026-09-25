"""§C.47 leg C, Stage 0 v2 — the PRE-REGISTERED Connect-4 go/no-go re-run after v1 (h43) read NOT_RUN, judged by
harness.stage0_v2.stage0_v2_report from evidence/c47_C0v2.json.gz (five fresh C_R nets, seeds 206-210). The proof
passes only on GO; the undecidable proof passes on UNDECIDED or NOT_RUN; STOP reads as the claim refuted. The
register pins this file, harness/stage0_v2.py and harness/stage0.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.stage0_v2 import stage0_v2_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c47_C0v2.json.gz",)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return stage0_v2_report(load_evidence(EVIDENCE / FILES[0]))


def test_c47_C0v2_the_direct_sibling_channel_has_headroom_on_connect4():
    assert _report()["C0"]["verdict"] == "go"


def test_c47_C0v2_is_undecidable():
    assert _report()["C0"]["verdict"] in {"undecided", "not_run"}
