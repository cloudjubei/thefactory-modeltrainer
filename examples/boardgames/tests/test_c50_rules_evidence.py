"""§C.50 E1 — the PRE-REGISTERED claims about written-rule playbooks, read from evidence/c50_E1_rules.json.gz
(scripts/playbook_measure.py). Every claim first checks the evidence was written by the registered measurement code
(fingerprint 1f08a0b61ab5: the script, harness/playbook.py, the solvers and samplers it reads). The register pins
this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c50_E1_rules.json.gz"
FILES = (FILE,)
MEASUREMENT_FP = "1f08a0b61ab5"
SETS = ("ttt_all", "ttt_own_p0", "ttt_own_p1", "c4_cache", "c4_sample")
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence():
    e = load_evidence(EVIDENCE / FILE)
    assert e["measurement_fingerprint"] == MEASUREMENT_FP, "not the registered measurement code"
    assert set(e["sets"]) == set(SETS), "not the registered sets"
    return e


def test_c50_E1_the_tactics_playbook_is_optimal_wherever_it_fires_on_every_set():
    for name in SETS:
        r = _evidence()["sets"][name]["tactics"]
        assert r["covered"] > 0 and r["correct"] == r["covered"], name


def test_c50_E1_newell_simon_plays_perfectly_from_the_start_as_either_player():
    for name in ("ttt_own_p0", "ttt_own_p1"):
        r = _evidence()["sets"][name]["newell_simon"]
        assert r["positions"] > 0 and r["covered"] == r["positions"] and r["correct"] == r["positions"], name
