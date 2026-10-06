"""§3.6 — finding registered AFTER the data, from evidence/c49_s1_handoff_probe.json.gz (h175, h176 refuted): the
hint does not steer the SAT search — hinted and cold runs on each position build constraints within 10% of each
other and run within 10% of the same number of rounds, so the solver leaves the hinted map within its first
counterexample rounds."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_s1_handoff_probe.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_s1_hinted_and_cold_sat_do_the_same_work():
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    by = {(r["index"], r["arm"]): r for r in rows}
    indices = {r["index"] for r in rows}
    assert len(indices) == 4 and all(r["status"] == "budget" for r in rows)
    for i in indices:
        hot, cold = by[(i, "hinted")], by[(i, "cold")]
        assert abs(hot["constraints"] - cold["constraints"]) <= 0.1 * cold["constraints"]
        assert abs(hot["iterations"] - cold["iterations"]) <= 0.1 * cold["iterations"]
