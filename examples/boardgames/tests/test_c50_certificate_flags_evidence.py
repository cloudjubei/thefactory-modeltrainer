"""§C.50 C1b — DESCRIPTIVE, registered after the data: what the Connect-4 certificates add beyond tactics, read from
evidence/c50_C1b_certificate_flags.json.gz (scripts/c4_certificate_pilot.py, the C1 pilot re-run with each flag
split into "a Black reply ends the game" and "by certificate"). The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c50_C1b_certificate_flags.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _sets() -> dict:
    e = load_evidence(EVIDENCE / FILE)
    assert e["contradictions"] == 0
    return e["sets"]


def _sum(rows: dict, key: str) -> int:
    return sum(v[key] for v in rows.values())


def test_c50_C1b_on_the_nets_trees_through_ply_8_no_position_is_certified_and_certificates_flag_under_1pct():
    cache = _sets()["cache"]
    assert _sum(cache, "certified") == 0 and _sum(cache, "non_winning") == 21175
    assert _sum(cache, "flagged") == 3008 and _sum(cache, "flagged_by_certificate") == 106
    assert _sum(cache, "flagged_by_certificate") / _sum(cache, "non_winning") < 0.01


def test_c50_C1b_certificates_grow_with_the_stones_on_the_board():
    rand = _sets()["random"]
    certified = [rand[b]["certified"] for b in ("20", "26", "32", "38")]
    assert certified == [2, 7, 12, 43] and all(v["positions"] == 150 for v in rand.values())
    by_cert = [rand[b]["flagged_by_certificate"] / rand[b]["non_winning"] for b in ("20", "26", "32", "38")]
    assert by_cert == sorted(by_cert) and by_cert[0] < 0.03 and by_cert[-1] > 0.35
