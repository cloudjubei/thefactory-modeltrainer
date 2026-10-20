"""§3.6 A6 — finding registered AFTER the data, from evidence/c49_a6.json.gz against evidence/c49_b1_control.json.gz:
a stricter acceptance bar makes the first complete build smaller but leaves the improvement pass less to choose from.
On the 15 positions both complete, A6's first complete builds take 156,348 bits against the control's 348,245 (smaller
on 13), but improvement cuts the control's to 93,033 and A6's only to 120,834 (larger on 12): at 30x the build accepts
leaves with exceptions that improvement then splits only where the split is smaller (259 kept of 437 tried), while 100x
splits at build time without comparing (129 of 239). Over all 16: table moves 510 -> 1,007, leaves 2,597 -> 4,686, maps
766 -> 1,199 (58,299 -> 76,113 bits), exceptions 25,698 -> 22,262 bits — 3,436 bits of exceptions saved for 17,814 more
in maps and 21,166 more in the rest of the strategy."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a6.json.gz"
FILES = (FILE, "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_a6_stricter_acceptance_builds_smaller_first_but_leaves_improvement_less_to_choose():
    arm = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    control = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_b1_control.json.gz")["roots"]}
    both = [i for i in arm if arm[i]["complete"] and control[i]["complete"]]

    def first(roots):
        return [roots[i]["improved"]["complete_bits"] for i in both]

    def final(roots):
        return [roots[i]["bits"]["nodes"] for i in both]

    def improvement(roots, key):
        return sum(roots[i]["improved"][key] for i in both)

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    assert len(both) == 15
    assert (sum(first(arm)), sum(first(control))) == (156_348, 348_245)
    assert sum(a < c for a, c in zip(first(arm), first(control))) == 13
    assert (sum(final(arm)), sum(final(control))) == (120_834, 93_033)
    assert sum(a > c for a, c in zip(final(arm), final(control))) == 12
    assert (improvement(arm, "kept"), improvement(arm, "tried")) == (129, 239)
    assert (improvement(control, "kept"), improvement(control, "tried")) == (259, 437)
    assert (total(arm, "moves"), total(control, "moves")) == (1_007, 510)
    assert (total(arm, "leaves"), total(control, "leaves")) == (4_686, 2_597)
    assert (total(arm, "maps"), total(control, "maps")) == (1_199, 766)
    assert (total(arm, "map_bits"), total(control, "map_bits")) == (76_113, 58_299)
    assert (total(arm, "exception_bits"), total(control, "exception_bits")) == (22_262, 25_698)
    assert (total(arm, "nodes"), total(control, "nodes")) == (140_404, 104_860)
