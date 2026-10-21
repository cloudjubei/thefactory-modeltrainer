"""§3.6 L6 — finding registered AFTER the data, from evidence/c49_l6.json.gz against evidence/c49_l5.json.gz: breaking
ties by tactics trades maps for exceptions and helps the light positions, not the heavy tail. More leaves need no map
(424 trivial against 281) and maps fall 784 -> 699 (59,697 -> 52,122 bits), but exceptions rise 2,262 -> 2,656
(28,764 -> 32,407 bits). F1's 16 heaviest of the 32 take 94,250 bits against 96,913, the other 16 take 8,039 against
12,104. Smaller on 19 positions, larger on 11, 2 tied — a sign test's two-sided p of 0.2, so the whole (0.94) is not
told apart from run-to-run swing."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l4 import SPEC as L4

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l6.json.gz"
FILES = (FILE, "c49_l5.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_l6_tactics_trade_maps_for_exceptions_and_help_the_light_positions_not_the_heavy_tail():
    arm = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    control = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_l5.json.gz")["roots"]}

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    def kind(roots, k):
        return sum(r["leaf_kinds"].get(k, 0) for r in roots.values())

    def part(roots, indices):
        return sum(roots[i]["bits"]["nodes"] for i in indices)

    assert (kind(arm, "trivial"), kind(control, "trivial")) == (424, 281)
    assert (total(arm, "maps"), total(control, "maps")) == (699, 784)
    assert (total(arm, "map_bits"), total(control, "map_bits")) == (52_122, 59_697)
    assert (total(arm, "exceptions"), total(control, "exceptions")) == (2_656, 2_262)
    assert (total(arm, "exception_bits"), total(control, "exception_bits")) == (32_407, 28_764)
    assert (part(arm, L4["indices"][:16]), part(control, L4["indices"][:16])) == (94_250, 96_913)
    assert (part(arm, L4["indices"][16:]), part(control, L4["indices"][16:])) == (8_039, 12_104)
    sizes = [(arm[i]["bits"]["nodes"], control[i]["bits"]["nodes"]) for i in arm]
    smaller, larger = sum(a < c for a, c in sizes), sum(a > c for a, c in sizes)
    assert (smaller, larger) == (19, 11)
    n = smaller + larger
    assert round(2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n, 2) == 0.2
