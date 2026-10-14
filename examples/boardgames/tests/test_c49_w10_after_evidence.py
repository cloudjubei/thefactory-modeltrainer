"""§3.6 W10 — finding registered AFTER the data, from evidence/c49_w10.json.gz against evidence/c49_w9.json.gz: most
of W10's cut is one position. #638 fell from 4,322 bits (70 leaves) to 356 — one root map with 17 exceptions, no
library leaf; the other 31 are 21% smaller together (6,757 bits against 8,586). Only 16 of 122 leaves reference the
library — three tiny roots covered outright at 8 bits each — so a library map as the searches' starting point (each
search starts from the most recent map) is as likely a cause as covering, and run-to-run variance cannot be ruled
out."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w10.json.gz"
FILES = (FILE, "c49_w9.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_w10_most_of_the_cut_is_one_position_and_few_leaves_use_the_library():
    w10 = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    w9 = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_w9.json.gz")["roots"]}
    assert (w9[638]["bits"]["nodes"], w10[638]["bits"]["nodes"]) == (4_322, 356)
    assert w10[638]["library_leaves"] == 0 and w10[638]["bits"]["leaves"] == 1
    rest = [i for i in w10 if i != 638]
    assert sum(w10[i]["bits"]["nodes"] for i in rest) == 6_757 and sum(w9[i]["bits"]["nodes"] for i in rest) == 8_586
    assert sum(r["library_leaves"] for r in w10.values()) == 16 and sum(r["bits"]["leaves"] for r in w10.values()) == 122
    assert sorted(i for i, r in w10.items() if r["library_leaves"] and r["bits"]["nodes"] == 8) == [70, 614, 669]
