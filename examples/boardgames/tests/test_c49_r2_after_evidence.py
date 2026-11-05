"""§3.6 R2 — finding registered AFTER the data, from evidence/c49_r2.json.gz against evidence/c49_g1.json.gz (B1's
control for the judge): counting references by first_use in the builder grows the tree. Cheaper references make
splitting pay more often, so the 16 positions take 2,173 leaves and 449 moves against G1's 1,513 and 283, and lean
less on exceptions (14,639 bits against 19,101); the extra tree outweighs them — 82,251 bits against G1's 74,556,
larger on 10 positions, smaller on 6 (sign test p = 0.45). The run's 2-hour wall-clock caps were shared with another
session's load: 17.6 hours of builds against G1's 14.7 for 1,093 searches against 1,138, one position at the cap."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g1 import _coded
from harness.floor_r2 import SPEC, r2_coded, r2_size

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_r2.json.gz", "c49_g1.json.gz", "c49_b1_control.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_r2_first_use_references_grow_the_tree_more_than_they_save():
    r2, g1, control = (load_evidence(EVIDENCE / f) for f in FILES)
    by_index = {r["index"]: r for r in g1["roots"]}
    pairs = [(r, by_index[r["index"]]) for r in r2["roots"]]
    assert len(pairs) == 16 and all(a["complete"] and b["complete"] for a, b in pairs)

    def total(key, side):
        return sum(p[side]["bits"][key] for p in pairs)

    assert [(total(k, 0), total(k, 1)) for k in ("leaves", "moves", "exception_bits")] == [
        (2_173, 1_513), (449, 283), (14_639, 19_101)]
    size = r2_size(r2, g1, control, SPEC)
    assert (size["bits"], size["control_bits"], size["verdict"]) == (82_251, 74_556, "refuted")
    coded = [(r2_coded(a), _coded(b)) for a, b in pairs]
    smaller, larger = sum(a < b for a, b in coded), sum(a > b for a, b in coded)
    assert (smaller, larger) == (6, 10)
    assert round(2 * sum(comb(16, k) for k in range(smaller + 1)) / 2 ** 16, 2) == 0.45
    hours = [round(sum(p[side]["build_seconds"] for p in pairs) / 3600, 1) for side in (0, 1)]
    assert hours == [17.6, 14.7]
    assert [sum(p[side]["searches"]["tried"] for p in pairs) for side in (0, 1)] == [1_093, 1_138]
    assert sum(a["build_seconds"] >= 7_190 for a, _ in pairs) == 1
