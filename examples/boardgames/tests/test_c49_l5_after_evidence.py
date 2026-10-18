"""§3.6 L5 — finding registered AFTER the data, from evidence/c49_l5.json.gz against L4's control
(evidence/c49_l4_control.json.gz), same 32 positions, both complete: reading the safe moves on unlevelled cells as a
last level shrinks the maps. Levelled cells fall from 17,276 to 12,497 and map bits from 76,858 to 59,697; the whole
takes 109,017 bits against 130,061 (0.84). It is smaller on 24 positions, larger on 7, equal on one (#9) — a sign test
puts 24 of 31 at p = 0.003, beyond the ~10% run-to-run swing (h282). The gain is in the heavy tail: F1's 16 heaviest
take 96,913 bits against 121,313 (0.80), while the 16 spread ones grow from 8,748 to 12,104, most of it one position
(#121, 3,429 against 373)."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l5 import SPEC

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l5.json.gz"
FILES = (FILE, "c49_l4_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_l5_unlevelled_cells_last_shrink_the_maps_and_the_heavy_tail():
    l5 = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    control = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_l4_control.json.gz")["roots"]}
    assert all(r["complete"] for r in l5.values()) and all(r["complete"] for r in control.values())

    def levelled(roots):
        return sum(len(m) for r in roots.values() for m in r["maps"][r["shared"] + 1:])

    assert (levelled(l5), levelled(control)) == (12_497, 17_276)
    assert (sum(r["bits"]["map_bits"] for r in l5.values()),
            sum(r["bits"]["map_bits"] for r in control.values())) == (59_697, 76_858)
    smaller = sum(1 for i in l5 if l5[i]["bits"]["nodes"] < control[i]["bits"]["nodes"])
    larger = sum(1 for i in l5 if l5[i]["bits"]["nodes"] > control[i]["bits"]["nodes"])
    assert (smaller, larger) == (24, 7)
    assert 2 * sum(comb(31, k) for k in range(24, 32)) / 2 ** 31 < 0.004
    heavy, spread = SPEC["indices"][:16], SPEC["indices"][16:]
    assert (sum(l5[i]["bits"]["nodes"] for i in heavy), sum(control[i]["bits"]["nodes"] for i in heavy)) == (96_913,
                                                                                                         121_313)
    assert (sum(l5[i]["bits"]["nodes"] for i in spread), sum(control[i]["bits"]["nodes"] for i in spread)) == (12_104,
                                                                                                           8_748)
    assert (l5[121]["bits"]["nodes"], control[121]["bits"]["nodes"]) == (3_429, 373)
