"""§3.6 B1 — finding registered AFTER the data, from evidence/c49_b1.json.gz against evidence/c49_b1_control.json.gz:
longer deeper searches make the heavy strategies larger by settling early for exception-heavy leaves. With 120 s the
builds stop splitting sooner — 196 table moves against 510, 1,061 leaves against 2,597 — and their maps cost less
(39,575 bits against 58,299), but they carry 9,094 exceptions against 1,972 (88,728 bits against 25,698): more time lets
a search near the root find a map that clears the 30x acceptance threshold with thousands of exceptions, where a split
would have been cheaper. The whole: 136,006 bits against 104,860, larger on 10 positions of 16."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_b1.json.gz"
FILES = (FILE, "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_b1_longer_searches_settle_early_for_exception_heavy_leaves():
    arm = load_evidence(EVIDENCE / FILE)["roots"]
    control = load_evidence(EVIDENCE / "c49_b1_control.json.gz")["roots"]

    def total(roots, key):
        return sum(r["bits"][key] for r in roots)

    assert (total(arm, "moves"), total(control, "moves")) == (196, 510)
    assert (total(arm, "leaves"), total(control, "leaves")) == (1_061, 2_597)
    assert (total(arm, "map_bits"), total(control, "map_bits")) == (39_575, 58_299)
    assert (total(arm, "exceptions"), total(control, "exceptions")) == (9_094, 1_972)
    assert (total(arm, "exception_bits"), total(control, "exception_bits")) == (88_728, 25_698)
    assert (total(arm, "nodes"), total(control, "nodes")) == (136_006, 104_860)
    c = {r["index"]: r["bits"]["nodes"] for r in control}
    assert sum(1 for r in arm if r["bits"]["nodes"] > c[r["index"]]) == 10
