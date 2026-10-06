"""§C.49 after H3 — PRE-REGISTERED claims on net cost against table cost as the certified depth grows, judged by
harness.depth_cost.depth_report from evidence/c49_depth_cost.json.gz (scripts/c4_depth_cost.py over H3's nets): the
nets' exception share falls from ply 8 to ply 12 of their own hybrid trees, pooled over seeds 481-483; and at depth 13
every one of those nets breaks even above 4 bits per weight. Each proof passes only on SUPPORTED; the undecidable
proofs pass on INCONCLUSIVE or NOT_RUN. The register pins this file and harness/depth_cost.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.depth_cost import SPEC, depth_report
from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_depth_cost.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return depth_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_depth_the_net_s_exception_share_falls_from_ply_8_to_ply_12():
    assert _report()["trend"]["verdict"] == "supported"


def test_c49_depth_the_trend_is_undecidable():
    assert _report()["trend"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_depth_every_deep_net_breaks_even_above_4_bits_a_weight_at_depth_13():
    assert _report()["break_even"]["verdict"] == "supported"


def test_c49_depth_the_break_even_is_undecidable():
    assert _report()["break_even"]["verdict"] in {"inconclusive", "not_run"}
