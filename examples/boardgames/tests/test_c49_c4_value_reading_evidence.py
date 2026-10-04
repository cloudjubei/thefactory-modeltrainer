"""§C.49 T11c — the PRE-REGISTERED claim that the value-aware stop would NOT fire on Connect-4 nets the solver does
not certify: on each of the six T10 final nets (none certified through 10 plies, h91) the value-aware reading at
delta 0.1 over its own first-player tree through 10 plies (T10's 200-sim search) is above zero. Read from
evidence/c49_T11c_c4_value_reading.json.gz (scripts/c4_value_reading_probe.py). The proof passes only on SUPPORTED;
the undecidable proof passes on INCONCLUSIVE. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T11c_c4_value_reading.json.gz"
FILES = (FILE,)
SEEDS = [361, 362, 363, 364, 365, 366]
CONFIG = {"depth": 10, "sims": 200, "deltas": [0.05, 0.1, 0.2]}
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _verdict() -> str:
    e = load_evidence(EVIDENCE / FILE)
    assert e["config"] == CONFIG
    rows = e["seeds"]
    if sorted(r["seed"] for r in rows) != SEEDS or not all(r["walked"] > 0 for r in rows):
        return "inconclusive"
    return "supported" if all(r["value_0.1"] > 0 for r in rows) else "refuted"


def test_c49_T11c_the_value_stop_reads_above_zero_on_every_uncertified_T10_net():
    assert _verdict() == "supported"


def test_c49_T11c_is_undecidable():
    assert _verdict() == "inconclusive"
