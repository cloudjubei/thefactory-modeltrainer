"""§3.6 S4 pilots — findings registered AFTER the data, on the two roots after S3's sample (1 h each, size-scored
local-search leaves): at a 4x acceptance threshold (evidence/c49_s4_pilot.json.gz) one root completes in 31 minutes
but exceptions are over 90% of its bits and the strategy is under 10x its table; at 10x
(evidence/c49_s4_pilot_accept10.json.gz) neither completes within the hour, and exceptions fall below 65% of the
partial strategies' bits."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_s4_pilot.json.gz", "c49_s4_pilot_accept10.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _roots(name, accept):
    e = load_evidence(EVIDENCE / name)
    assert e["builder"]["accept"] == accept and e["builder"]["seconds"] == 3600.0 and len(e["roots"]) == 2
    return e["roots"]


def test_c49_s4_pilot_at_4x_completes_a_root_whose_bits_are_mostly_exceptions():
    done = [r for r in _roots(FILES[0], 4) if r["complete"]]
    assert len(done) == 1 and done[0]["checked"] and done[0]["moves_win"] and done[0]["leaves_certified"]
    r = done[0]
    assert r["bits"]["exception_bits"] > 0.9 * r["bits"]["nodes"] and 3 * r["own_positions"] / r["bits"]["nodes"] < 10


def test_c49_s4_pilot_at_10x_completes_nothing_in_an_hour_but_halves_the_exception_share():
    roots = _roots(FILES[1], 10)
    assert not any(r["complete"] for r in roots)
    assert all(r["bits"]["exception_bits"] < 0.65 * r["bits"]["nodes"] for r in roots)
