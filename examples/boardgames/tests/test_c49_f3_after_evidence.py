"""§3.6 F3 — finding registered AFTER the data, from evidence/c49_f3.json.gz against evidence/c49_f2.json.gz: L6's
tie-break saves the whole game 33,732 bits (422,227 against 455,959, 0.93) and ten positions hold 32,763 of them —
smaller on 335 of the 671, larger on 246, equal on 90 (sign test p = 0.0003), the rest cancelling. Leaves 9,495 ->
8,722, own maps 2,801 -> 2,553, exceptions 8,354 -> 8,612; 297 CPU-hours against F2's 216."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_f3.json.gz"
FILES = (FILE, "c49_f2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_f3_the_tie_break_s_saving_is_ten_positions():
    arm = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    control = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_f2.json.gz")["roots"]}

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    assert (total(arm, "nodes"), total(control, "nodes")) == (412_534, 446_266)
    changes = sorted(arm[i]["bits"]["nodes"] - control[i]["bits"]["nodes"] for i in arm)
    assert (sum(changes), sum(changes[:10])) == (-33_732, -32_763)
    assert (total(arm, "leaves"), total(control, "leaves")) == (8_722, 9_495)
    assert (total(arm, "maps"), total(control, "maps")) == (2_553, 2_801)
    assert (total(arm, "exceptions"), total(control, "exceptions")) == (8_612, 8_354)
    hours = [round(sum(r["build_seconds"] for r in roots.values()) / 3600) for roots in (arm, control)]
    assert hours == [297, 216]
    smaller, larger = sum(c < 0 for c in changes), sum(c > 0 for c in changes)
    assert (smaller, larger, len(changes) - smaller - larger) == (335, 246, 90)
    n = smaller + larger
    assert round(2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n, 4) == 0.0003
