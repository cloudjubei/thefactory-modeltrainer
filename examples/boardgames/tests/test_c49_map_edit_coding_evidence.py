"""§3.6 — dead end registered AFTER the data, from evidence/c49_a4.json.gz: the maps of a strategy are not near-copies
of one another, though each search starts from the most recent map. Coded as an edit of the map before it — how many
cells differ (in the bits to count to its empty cells), which of the 42 cells, and each one's level in 3 bits — A4's
215 maps would take 20,354 bits against 17,259 coded whole (harness.steady_state.map_bits): 18% more."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.steady_state import map_bits

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_a4.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def _edit_bits(levels: dict, previous: dict, empty: int) -> int:
    differ = sum(1 for c in set(levels) | set(previous) if levels.get(c) != previous.get(c))
    return math.ceil(math.log2(empty + 1)) + math.ceil(math.log2(math.comb(42, differ))) + 3 * differ


def test_c49_maps_coded_as_edits_of_the_map_before_cost_more_than_coded_whole():
    whole = edits = count = 0
    for r in load_evidence(EVIDENCE / FILE)["roots"]:
        maps = [{int(c): k for c, k in m.items()} for m in r["maps"]]
        for i in range(1, len(maps)):
            whole += map_bits(maps[i], r["map_empty"][i], 3)
            edits += _edit_bits(maps[i], maps[i - 1], r["map_empty"][i])
            count += 1
    assert (count, whole, edits) == (215, 17_259, 20_354)
