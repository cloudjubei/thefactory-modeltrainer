"""§3.6 X2's probe — finding registered AFTER the data, from evidence/c49_x2_probe.json.gz (scripts/c4_mirror_probe.py):
walking Z1's 51 rebuilds from their roots with no build, the 2,598 maps with their cells flipped cover none of the 343
move nodes; the same maps read on the mirrored board cover 18, in 7 rebuilds, and those subtrees hold 260 of the 2,245
nodes, 166 of them in #2's 215. A map's rule breaks ties by the lowest move, so flipping a map is not reading it
mirrored."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "evidence" / "c49_x2_probe.json.gz"
pytestmark = pytest.mark.skipif(
    not FILE.exists(), reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proof")


def test_c49_x2_only_the_mirrored_reading_covers_move_nodes():
    e = load_evidence(FILE)
    rows = e["roots"]
    assert (len(rows), e["library"]) == (51, 2_598)
    assert (sum(r["moves"] for r in rows), sum(r["nodes"] for r in rows)) == (343, 2_245)
    assert sum(r["flipped"]["hits"] for r in rows) == 0
    assert sum(r["mirrored"]["hits"] for r in rows) == 18
    assert sum(r["mirrored"]["nodes_cut"] for r in rows) == 260
    assert sum(r["mirrored"]["hits"] > 0 for r in rows) == 7
    assert max((r["mirrored"]["nodes_cut"], r["nodes"], r["index"]) for r in rows) == (166, 215, 2)
