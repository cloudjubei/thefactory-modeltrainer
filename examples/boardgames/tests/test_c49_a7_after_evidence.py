"""§3.6 A7 — finding registered AFTER the data, from evidence/c49_a7.json.gz against evidence/c49_b1_control.json.gz:
the 10x bar's 7.5% is two positions, not the bar — the strategies are smaller on 7 positions and larger on 9 (sign
test p = 0.80), and two of them (#212: 9,518 -> 3,281; #113: 12,241 -> 8,427) hold 10,051 of the 7,893 bits saved.
Leaves 2,597 -> 2,497, maps 766 -> 696 (58,299 -> 54,167 bits), exceptions 1,972 -> 1,800 (25,698 -> 23,439 bits)."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a7.json.gz"
FILES = (FILE, "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_a7_the_saving_is_two_positions_not_the_bar():
    arm = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    control = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_b1_control.json.gz")["roots"]}

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    assert (total(arm, "nodes"), total(control, "nodes")) == (96_967, 104_860)
    assert (total(arm, "leaves"), total(control, "leaves")) == (2_497, 2_597)
    assert (total(arm, "maps"), total(control, "maps")) == (696, 766)
    assert (total(arm, "map_bits"), total(control, "map_bits")) == (54_167, 58_299)
    assert (total(arm, "exceptions"), total(control, "exceptions")) == (1_800, 1_972)
    assert (total(arm, "exception_bits"), total(control, "exception_bits")) == (23_439, 25_698)
    sizes = {i: (arm[i]["bits"]["nodes"], control[i]["bits"]["nodes"]) for i in arm}
    assert (sizes[212], sizes[113]) == ((3_281, 9_518), (8_427, 12_241))
    assert sum(c - a for i, (a, c) in sizes.items() if i in (212, 113)) == 10_051
    smaller, larger = sum(a < c for a, c in sizes.values()), sum(a > c for a, c in sizes.values())
    assert (smaller, larger) == (7, 9)
    n = smaller + larger
    assert round(2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n, 2) == 0.80
