"""§3.6 W10 — PRE-REGISTERED claims on W9's 32 fresh frontier positions rebuilt with L2's rule and pipeline from a
library of the 90 maps L2 found on W6's positions, judged by harness.floor_w10 from evidence/c49_w10.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_w10): at least 28 of the 32 complete within 2 hours; the whole game,
the library's 8,618 bits paid once, projects to at most four fifths of W9's 271,740 bits; and at least a fifth of the
leaves use a library map."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_w10 import SPEC, w10_library, w10_projection, w10_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w10.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_w10_28_of_32_complete_within_2_hours():
    assert w10_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_w10_completion_is_undecidable():
    assert w10_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_w10_a_shared_library_cuts_the_whole_game_by_a_fifth():
    assert w10_projection(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w10_the_projection_against_w9_is_undecidable():
    assert w10_projection(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_w10_a_fifth_of_the_leaves_use_the_library():
    assert w10_library(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_w10_library_use_is_undecidable():
    assert w10_library(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
