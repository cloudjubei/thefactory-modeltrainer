"""§C.49 T20 — the PRE-REGISTERED readings of whether the Connect-4 base process is still learning at iteration 20:
the base pilot recipe run for 60 iterations on seeds 471-473, the net scored on the fixed position set after every
training pass, the share at passes 56-60 against passes 16-20 — at plies 0-4 (h127) and over all positions (h128).
Judged by harness.curve_diagnosis.curve_report from evidence/c49_T20_s471.json.gz, c49_T20_s472.json.gz and
c49_T20_s473.json.gz (scripts/c4_solver_free.py --curve, one seed each). Each proof passes only on SUPPORTED; each
undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file and harness/curve_diagnosis.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.curve_diagnosis import SPEC, curve_report
from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = tuple(f"c49_T20_s{s}.json.gz" for s in SPEC["seeds"])
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report() -> dict:
    return curve_report([load_evidence(EVIDENCE / f) for f in FILES], SPEC)


def test_c49_T20_the_opening_is_still_improving_at_iteration_20():
    assert _report()["opening"]["verdict"] == "supported"


def test_c49_T20_the_opening_reading_is_undecidable():
    assert _report()["opening"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_T20_play_overall_is_still_improving_at_iteration_20():
    assert _report()["overall"]["verdict"] == "supported"


def test_c49_T20_the_overall_reading_is_undecidable():
    assert _report()["overall"]["verdict"] in {"inconclusive", "not_run"}
