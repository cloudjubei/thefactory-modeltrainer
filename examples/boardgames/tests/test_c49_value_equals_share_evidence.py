"""§C.49 T11d — DESCRIPTIVE, registered after the data: on the six T10 Connect-4 nets the value-aware reading at
delta 0.05 and 0.1 equals the share reading exactly, and delta 0.2 removes at most a fifth of the disagreements — the
disagreements are between moves the search itself values more than 0.1 apart. Read from
evidence/c49_T11c_c4_value_reading.json.gz. The register pins this file."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T11c_c4_value_reading.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_T11d_the_value_reading_is_the_share_reading_at_delta_0_1_on_every_T10_net():
    rows = load_evidence(EVIDENCE / FILE)["seeds"]
    assert [r["share"] for r in rows] == [76, 71, 17, 59, 53, 78]
    assert all(r["value_0.05"] == r["value_0.1"] == r["share"] for r in rows)
    assert all(0.8 * r["share"] <= r["value_0.2"] <= r["share"] for r in rows)
