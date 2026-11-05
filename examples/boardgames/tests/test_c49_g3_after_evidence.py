"""§3.6 G3 — finding registered AFTER the data, from evidence/c49_g3.json.gz against evidence/c49_g2.json.gz: the
third pass still shrinks almost every position — smaller on 562 of the 671, larger on 103, equal on 6 (sign test
p < 1e-70) — but by 0.955 of G2's whole game, not a tenth: references by first_use the strategies take 176,750 bits
against G2's 202,670, while the 1,207 library maps they play cost 102,815 against G2's 90,098. Of 2,689 library
plays, 1,029 are F2's maps G2 kept and 1,660 G2's own. Leaves 6,841 -> 6,330, own maps 931 -> 709; 151 CPU-hours."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g1 import _coded
from harness.floor_g2 import SPEC as G2, g2_size
from harness.floor_g3 import SPEC, g3_library, g3_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_g3.json.gz"
FILES = (FILE, "c49_g2.json.gz", "c49_f2.json.gz")
OPENING = 204 * 4 + 259
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_g3_the_third_pass_trades_strategy_bits_for_library_bits():
    e, g2, f2 = (load_evidence(EVIDENCE / f) for f in FILES)
    arm = {r["index"]: r for r in e["roots"]}
    control = {r["index"]: r for r in g2["roots"]}
    library = g3_library(g2, f2)

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    coded = (sum(_coded(r) for r in arm.values()), sum(_coded(r) for r in control.values()))
    assert coded == (176_750, 202_670)
    assert g3_size(e, g2, f2, SPEC)["bits"] - coded[0] - OPENING == 102_815
    assert g2_size(g2, f2, G2)["bits"] - coded[1] - OPENING == 90_098
    plays = [m for r in arm.values() for m in {n["leaf"] for _b, _t, n in r["nodes"]
                                                if 1 <= n.get("leaf", 0) <= len(library)}]
    assert (len(set(plays)), len(plays)) == (1_207, 2_689)
    assert (sum(m <= 1_099 for m in plays), sum(m > 1_099 for m in plays)) == (1_029, 1_660)
    assert (total(arm, "leaves"), total(control, "leaves")) == (6_330, 6_841)
    assert (total(arm, "maps"), total(control, "maps")) == (709, 931)
    assert (total(arm, "exceptions"), total(control, "exceptions")) == (3_359, 3_581)
    assert round(sum(r["build_seconds"] for r in arm.values()) / 3600) == 151
    sizes = [(_coded(arm[i]), _coded(control[i])) for i in arm]
    smaller, larger = sum(a < c for a, c in sizes), sum(a > c for a, c in sizes)
    assert (smaller, larger, len(sizes) - smaller - larger) == (562, 103, 6)
    n = smaller + larger
    assert 2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n < 1e-70
