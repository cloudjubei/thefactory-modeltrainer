"""§3.6 G1 — finding registered AFTER the data, from evidence/c49_g1.json.gz against evidence/c49_b1_control.json.gz:
the library's quarter is every position, not a few — smaller on 15 of the 16, larger on 1 (sign test p = 0.0005).
References by first_use, the strategies take 52,860 bits against the control's 99,319 and the 277 library maps they
play 21,696 more; leaves 2,597 -> 1,513, own maps 766 -> 252, exceptions 1,972 -> 1,569. Of the 342 times a position
plays a library map, 212 are maps the control found on that same position and 130 another's."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g1 import _coded, g1_library, g1_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_g1.json.gz"
FILES = (FILE, "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_g1_the_library_shrinks_every_position_but_one():
    e = load_evidence(EVIDENCE / FILE)
    c = load_evidence(EVIDENCE / "c49_b1_control.json.gz")
    arm = {r["index"]: r for r in e["roots"]}
    control = {r["index"]: r for r in c["roots"]}
    library = g1_library(c)
    found_on = [r["index"] for r in c["roots"] for _ in r["found_maps"]]

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    assert (sum(_coded(r) for r in arm.values()), sum(_coded(r) for r in control.values())) == (52_860, 99_319)
    plays = {i: {n["leaf"] for _b, _t, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}
             for i, r in arm.items()}
    assert len(set().union(*plays.values())) == 277 and g1_size(e, c)["bits"] - 52_860 == 21_696
    assert sum(found_on[m - 1] == i for i, used in plays.items() for m in used) == 212
    assert sum(found_on[m - 1] != i for i, used in plays.items() for m in used) == 130
    assert (total(arm, "leaves"), total(control, "leaves")) == (1_513, 2_597)
    assert (total(arm, "maps"), total(control, "maps")) == (252, 766)
    assert (total(arm, "exceptions"), total(control, "exceptions")) == (1_569, 1_972)
    sizes = [(_coded(arm[i]), _coded(control[i])) for i in arm]
    smaller, larger = sum(a < c for a, c in sizes), sum(a > c for a, c in sizes)
    assert (smaller, larger) == (15, 1)
    n = smaller + larger
    assert round(2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n, 4) == 0.0005
