"""§3.6 K0 — claim registered before the run, judged by harness.floor_k0 from evidence/c49_k0.json.gz
(scripts/c4_phase_pilot.py --spec harness.floor_k0): at positions where L6's heavy strategies split, the 30-s leaf
search with two-phase maps finds a smaller leaf at least twice as often as a larger one."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_k0 import SPEC, k0_sign

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_k0.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_k0_two_phase_searches_find_smaller_leaves_twice_as_often_as_larger():
    assert k0_sign(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_k0_the_sign_is_undecidable():
    assert k0_sign(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
