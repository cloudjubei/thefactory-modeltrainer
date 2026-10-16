"""§3.6 M1 — PRE-REGISTERED claim on F1's 3,196 maps coded with the cheapest of harness.map_coding's exact codes plus
a selector, judged by harness.floor_m1 from evidence/c49_m1.json.gz (scripts/c4_map_coding.py --spec harness.floor_m1):
the cheaper codes save at least a tenth of F1's 292,054 map bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_m1 import SPEC, m1_saving

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_m1.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_m1_cheaper_exact_codes_save_a_tenth_of_f1_s_map_bits():
    assert m1_saving(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_m1_the_saving_is_undecidable():
    assert m1_saving(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
