"""§3.6 F1 — PRE-REGISTERED claims on the whole game built: every one of the opening's 671 frontier positions, with L2's
rule and pipeline, in waves from a growing library seeded with L2's 90 maps, judged by harness.floor_f1 from
evidence/c49_f1.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_f1): every position completes within 2
hours; the certified whole-game first-player strategy — the opening, the seed library once and every position's own
bits — takes no more than W9's projection of 271,740 bits; and it is smaller than the trained net's ~660K."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_f1 import SPEC, f1_complete, f1_net, f1_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_f1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_f1_every_frontier_position_completes():
    assert f1_complete(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_f1_completion_is_undecidable():
    assert f1_complete(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_f1_the_whole_strategy_is_within_w9_s_projection():
    assert f1_size(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_f1_the_whole_strategy_s_size_is_undecidable():
    assert f1_size(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_f1_the_whole_strategy_is_smaller_than_the_net():
    assert f1_net(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_f1_the_comparison_with_the_net_is_undecidable():
    assert f1_net(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
