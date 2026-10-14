"""§3.6 A2 — finding registered AFTER the data, from evidence/c49_a2.json.gz against evidence/c49_a1.json.gz and
evidence/c49_w6.json.gz: choosing between a leaf with exceptions and the split below it by size makes smaller
strategies than either fixed threshold where it finishes — #95 3,481 bits (best threshold build 5,307), #535 6,637
(12,857) — but recursively it is too slow for 2 hours: #565's split never finished (the leaf was kept, W6's 3,877
exactly) and #594 did not finish at all."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a2.json.gz"
FILES = (FILE, "c49_a1.json.gz", "c49_w6.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _bits(file: str) -> dict:
    return {r["index"]: (r["complete"], r["bits"]["nodes"], r["build_seconds"])
            for r in load_evidence(EVIDENCE / file)["roots"]}


def test_c49_a2_choosing_by_size_beats_both_thresholds_where_it_finishes_but_runs_out_of_time():
    a2, a1, w6 = _bits(FILE), _bits("c49_a1.json.gz"), _bits("c49_w6.json.gz")
    best = {i: min(a1[i][1], w6[i][1]) for i in a2}
    for i in (95, 535):
        assert a2[i][0] and a2[i][1] < 0.7 * best[i]
    assert a2[565][0] and a2[565][1] == w6[565][1] and a2[565][2] >= 7200
    assert not a2[594][0] and a2[594][2] >= 7200
    finished = [i for i in a2 if a2[i][0]]
    assert sum(a2[i][1] for i in finished) == 13_995 and sum(best[i] for i in finished) == 22_041
