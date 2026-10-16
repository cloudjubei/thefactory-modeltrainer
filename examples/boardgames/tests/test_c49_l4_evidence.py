"""§3.6 L4 — PRE-REGISTERED claims on 32 of F1's positions (its 16 heaviest, 16 spread over the rest) built with L2's
rule and pipeline preferring forcing moves, against a control built the same way without, judged by harness.floor_l4
from evidence/c49_l4.json.gz and evidence/c49_l4_control.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_l4,
then --spec harness.floor_l4_control): at least 28 of the 32 complete within 2 hours preferring forcing moves, and
they take no more than four fifths of the control's bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l4 import SPEC, l4_report, l4_size
from harness.floor_l4_control import SPEC as CONTROL

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l4.json.gz"
FILES = (FILE, "c49_l4_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=2)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


def _size() -> dict:
    return l4_size(_evidence(), _evidence("c49_l4_control.json.gz"), SPEC, CONTROL)


def test_c49_l4_28_of_32_complete_within_2_hours_preferring_forcing_moves():
    assert l4_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_l4_completion_is_undecidable():
    assert l4_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_l4_preferring_forcing_moves_makes_the_strategies_a_fifth_smaller():
    assert _size()["verdict"] == "supported"


def test_c49_l4_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
