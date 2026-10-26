"""§3.6 G2 — finding registered AFTER the data, from evidence/c49_g2.json.gz against evidence/c49_f2.json.gz: the
second pass shrinks the whole game, not a few positions — smaller on 540 of the 671, larger on 102, equal on 29 (sign
test p < 1e-70). References by first_use, the strategies take 202,670 bits against F2's 388,502 and the 1,099 of F2's
2,801 found maps they play 90,098 more; leaves 9,495 -> 6,841, own maps 2,801 -> 931, exceptions 8,354 -> 3,581. Of
the 2,661 times a position plays a library map, 741 are maps F2 found on that same position and 1,920 another's."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g1 import _coded, g1_library
from harness.floor_g2 import SPEC, g2_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_g2.json.gz"
FILES = (FILE, "c49_f2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_g2_the_second_pass_shrinks_most_positions_and_shares_maps_across_them():
    e = load_evidence(EVIDENCE / FILE)
    f2 = load_evidence(EVIDENCE / "c49_f2.json.gz")
    arm = {r["index"]: r for r in e["roots"]}
    control = {r["index"]: r for r in f2["roots"]}
    library = g1_library(f2)
    found_on = [r["index"] for r in f2["roots"] for _ in r["found_maps"]]

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    coded = sum(_coded(r) for r in arm.values())
    assert (coded, sum(_coded(r) for r in control.values())) == (202_670, 388_502)
    plays = {i: {n["leaf"] for _b, _t, n in r["nodes"] if 1 <= n.get("leaf", 0) <= len(library)}
             for i, r in arm.items()}
    assert len(library) == 2_801 and len(set().union(*plays.values())) == 1_099
    assert g2_size(e, f2, SPEC)["bits"] - coded - (204 * 4 + 259) == 90_098
    assert sum(found_on[m - 1] == i for i, used in plays.items() for m in used) == 741
    assert sum(found_on[m - 1] != i for i, used in plays.items() for m in used) == 1_920
    assert (total(arm, "leaves"), total(control, "leaves")) == (6_841, 9_495)
    assert (total(arm, "maps"), total(control, "maps")) == (931, 2_801)
    assert (total(arm, "exceptions"), total(control, "exceptions")) == (3_581, 8_354)
    sizes = [(_coded(arm[i]), _coded(control[i])) for i in arm]
    smaller, larger = sum(a < c for a, c in sizes), sum(a > c for a, c in sizes)
    assert (smaller, larger, len(sizes) - smaller - larger) == (540, 102, 29)
    n = smaller + larger
    assert 2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n < 1e-70
