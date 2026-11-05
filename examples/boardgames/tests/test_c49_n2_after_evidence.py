"""§3.6 N2 — finding registered AFTER the data, from evidence/c49_n2.json.gz against evidence/c49_l6.json.gz and
evidence/c49_l5.json.gz: breaking ties by N1's net does not shrink the strategies. Four positions (#113, #207, #218,
#240) were built while the machine was swapping and ran 22,716-22,721 s against the 2-hour cap; on the other 28, all
complete, the net's strategies take 68,693 bits against tactics' (L6) 55,809 and L5's 64,648 — smaller than L6 on 11,
larger on 15, equal on 2 (sign test p = 0.56). Even were the four as small as L6's, N2 would take 115,173 bits, above
L6's 102,289, so h328 is refuted whatever the overload cost; 45.3 hours of builds."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_n2.json.gz", "c49_l6.json.gz", "c49_l5.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_n2_the_net_breaks_ties_no_better_than_tactics():
    n2, l6, l5 = (load_evidence(EVIDENCE / f) for f in FILES)
    tactics = {r["index"]: r["bits"]["nodes"] for r in l6["roots"]}
    plain = {r["index"]: r["bits"]["nodes"] for r in l5["roots"]}
    overloaded = [r for r in n2["roots"] if r["build_seconds"] > 3 * 3600]
    quiet = [r for r in n2["roots"] if r["build_seconds"] <= 3 * 3600]
    assert sorted(r["index"] for r in overloaded) == [113, 207, 218, 240]
    assert [round(r["build_seconds"]) for r in sorted(overloaded, key=lambda r: r["index"])] == [22717, 22716,
                                                                                                22717, 22721]
    assert len(quiet) == 28 and all(r["complete"] for r in quiet)
    bits = [(r["bits"]["nodes"], tactics[r["index"]]) for r in quiet]
    assert (sum(a for a, _ in bits), sum(b for _, b in bits), sum(plain[r["index"]] for r in quiet)) == (
        68_693, 55_809, 64_648)
    smaller, larger = sum(a < b for a, b in bits), sum(a > b for a, b in bits)
    assert (smaller, larger, len(bits) - smaller - larger) == (11, 15, 2)
    n = smaller + larger
    assert round(2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n, 2) == 0.56
    best_case = sum(a for a, _ in bits) + sum(tactics[r["index"]] for r in overloaded)
    assert best_case == 115_173 > sum(tactics.values()) == 102_289
    assert round(sum(r["build_seconds"] for r in n2["roots"]) / 3600, 1) == 45.3
