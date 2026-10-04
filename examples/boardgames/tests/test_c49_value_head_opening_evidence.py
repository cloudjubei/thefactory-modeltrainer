"""§C.49 D3b — DESCRIPTIVE, registered after the data: where D3's value-head readings concentrate. Read from
evidence/c49_D3_value_head.json.gz through harness.value_diagnosis.d3_report. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.value_diagnosis import D3_SPEC, d3_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_D3_value_head.json.gz"
FILES = (FILE,)
CENTRE = 3
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence():
    return load_evidence(EVIDENCE / FILE)


def test_c49_D3b_the_value_head_backs_opening_errors_more_than_deeper_ones():
    by_ply = d3_report(_evidence(), D3_SPEC)["by_ply"]
    opening = [by_ply[p] for p in ("0", "2", "4")]
    deeper = [by_ply[p] for p in ("6", "8")]
    assert (sum(r["backed"] for r in opening), sum(r["errors"] for r in opening)) == (104, 182)
    assert (sum(r["backed"] for r in deeper), sum(r["errors"] for r in deeper)) == (3026, 7247)


def test_c49_D3b_at_the_empty_board_every_value_head_rates_the_winning_centre_below_two_other_moves():
    roots = [r for r in _evidence()["positions"] if r["ply"] == 0]
    below = []
    for r in roots:
        q = {int(a): v for a, v in r["q_hat"].items()}
        assert {a for a, v in r["values"].items() if v == 1} == {str(CENTRE)}
        below.append(sum(1 for a, v in q.items() if a != CENTRE and v > q[CENTRE]))
    assert len(roots) == 6 and all(n >= 2 for n in below) and sum(1 for n in below if n == 6) == 3
