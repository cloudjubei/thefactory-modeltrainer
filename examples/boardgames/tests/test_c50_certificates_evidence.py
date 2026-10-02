"""§C.50 C1 — the PRE-REGISTERED claims on the Allis-style Connect-4 value certificates (harness/c4_certificates.py),
read from evidence/c50_C1_certificates.json.gz (scripts/c4_certificate_pilot.py):

  h98  SOUND: no certified position is a White win and no flagged move wins, over the label cache, the random-play
       bands and the D2 opening; judged only when at least 100 moves were flagged.
  h99  USEFUL FOR THE OPENING: at plies 0-4 the certificates flag at least half of the search's non-optimal
       preferred White moves (pooled over the D2 budgets and seeds); judged only on at least 20 such moves.

Each proof passes only on SUPPORTED; each undecidable proof passes on INCONCLUSIVE. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c50_C1_certificates.json.gz"
FILES = (FILE,)
CONFIG = {"bands": [20, 26, 32, 38], "per_band": 150, "seed": 98, "opening_max_ply": 4}
MIN_FLAGGED = 100
MIN_NON_OPTIMAL = 20
USEFUL_SHARE = 0.5
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    e = load_evidence(EVIDENCE / FILE)
    assert e["config"] == CONFIG
    return e


def _sound() -> str:
    e = _evidence()
    sets = e["sets"]
    flagged = sum(v["flagged"] for name in ("cache", "random") for v in sets[name].values())
    if e["contradictions"]:
        return "refuted"
    return "supported" if flagged >= MIN_FLAGGED else "inconclusive"


def _useful() -> str:
    opening = _evidence()["sets"]["opening"]
    non_optimal = sum(v["non_optimal"] for budget in opening.values() for v in budget.values())
    flagged = sum(v["flagged"] for budget in opening.values() for v in budget.values())
    if non_optimal < MIN_NON_OPTIMAL:
        return "inconclusive"
    return "supported" if flagged >= USEFUL_SHARE * non_optimal else "refuted"


def test_c50_C1_certificates_never_contradict_the_solver():
    assert _sound() == "supported"


def test_c50_C1_soundness_is_undecidable():
    assert _sound() == "inconclusive"


def test_c50_C1_certificates_flag_at_least_half_the_search_s_non_optimal_opening_moves():
    assert _useful() == "supported"


def test_c50_C1_opening_usefulness_is_undecidable():
    assert _useful() == "inconclusive"
