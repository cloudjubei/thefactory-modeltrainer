"""§3.6 L3 — finding registered AFTER the data, from evidence/c49_l3.json.gz against evidence/c49_l2.json.gz:
two-ply safety trades exceptions for maps. Exceptions fall from 34% of L2's bits to 16% of L3's, but the strategies
need 118 maps against 90 and come out 4.6% larger in all (15,343 bits against 14,666) — smaller on 13 positions,
larger on 9 — with the builds a fifth slower."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l3.json.gz"
FILES = (FILE, "c49_l2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _share(roots, key):
    return sum(r["bits"][key] for r in roots) / sum(r["bits"]["nodes"] for r in roots)


def test_c49_l3_two_ply_safety_trades_exceptions_for_maps():
    l3 = load_evidence(EVIDENCE / FILE)["roots"]
    l2 = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_l2.json.gz")["roots"]}
    assert 0.33 < _share(l2.values(), "exception_bits") < 0.35 and 0.15 < _share(l3, "exception_bits") < 0.16
    assert sum(r["bits"]["maps"] for r in l3) == 118 and sum(r["bits"]["maps"] for r in l2.values()) == 90
    assert sum(r["bits"]["nodes"] for r in l3) == 15_343
    assert sum(r["bits"]["nodes"] < l2[r["index"]]["bits"]["nodes"] for r in l3) == 13
    assert sum(r["bits"]["nodes"] > l2[r["index"]]["bits"]["nodes"] for r in l3) == 9
    assert sum(r["build_seconds"] for r in l3) > 1.15 * sum(r["build_seconds"] for r in l2.values())
