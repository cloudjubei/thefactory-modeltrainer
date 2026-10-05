"""§C.49 H1b — DESCRIPTIVE, registered after the data: the hybrid's trade-off curve on the ten base-recipe nets — how
much of each opening table the net needs overridden, and how often it is wrong on the ply it carries. Read from
evidence/c49_H1_hybrid.json.gz. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_H1_hybrid.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _pooled() -> dict:
    nets = load_evidence(EVIDENCE / FILE)["nets"]
    out = {}
    for h in ("3", "5", "7"):
        r = [n["horizons"][h] for n in nets]
        out[h] = {"entries": sum(x["entries"] for x in r), "tabled": sum(x["positions"] for x in r),
                  "carried": sum(x["nodes"]["player"] - x["positions"] for x in r),
                  "failures": sum(x["failures"] for x in r), "certified": sum(1 for x in r if x["certified"])}
    return out


def test_c49_H1b_the_table_overrides_a_falling_share_of_its_positions():
    p = _pooled()
    assert [(p[h]["entries"], p[h]["tabled"]) for h in ("3", "5", "7")] == [(40, 80), (149, 563), (492, 3387)]


def test_c49_H1b_the_net_is_wrong_at_about_a_fifth_of_ply_4_and_an_eighth_of_plies_6_and_8_and_no_hybrid_certifies():
    p = _pooled()
    assert [(p[h]["failures"], p[h]["carried"]) for h in ("3", "5", "7")] == [(109, 483), (343, 2824), (1441, 12235)]
    assert all(p[h]["certified"] == 0 for h in p)
