"""§3.6 — finding registered AFTER the data, from evidence/c49_map_reuse.json.gz (scripts/c4_map_reuse.py): under L2's
rule maps carry across the frontier in part. Of the 223 leaves of W9's 32 strategies (fresh positions), 77 — 35% —
are won from by one of the 90 maps L2 found on W6's 32 positions, none of the 25 leaves with exceptions; 3 of W9's
32 roots are covered outright (tiny ones, 177 bits of 12,908)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_map_reuse.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_a_third_of_fresh_leaves_are_won_by_maps_found_elsewhere():
    e = load_evidence(EVIDENCE / FILE)
    rows = e["rows"]
    assert (e["spec"], e["maps_from"], e["targets"], e["maps"]) == ("harness.floor_l2", "c49_l2.json.gz",
                                                                     "c49_w9.json.gz", 90)
    assert len(rows) == 32 and sum(r["leaves"] for r in rows) == 223 and sum(r["leaves_covered"] for r in rows) == 77
    assert sum(r["excepted"] for r in rows) == 25 and sum(r["excepted_covered"] for r in rows) == 0
    covered = [r for r in rows if r["root_covered"]]
    assert len(covered) == 3 and sum(r["bits"] for r in covered) == 177
