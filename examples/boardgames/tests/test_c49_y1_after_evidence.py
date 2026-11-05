"""§3.6 Y1 — finding registered AFTER the data, from evidence/c49_weakc4.json.gz: WeakC4's edge is its tree, not its
language. Walked from the empty board (commit 51dc838), its strategy has 1,431 first-player positions: 251 table moves,
1,009 leaves on a diagram (from ply 4: 18 at ply 4, 65 at ply 6) and 171 won at once, with 745 maps in our cell-indexed
language, 377 of them mirrored uses of its 384 diagrams. Our "exact" rule reads 1,003 of the 1,009 as they are (h332).
In our code the whole solution takes 123,881 bits, 0.44 of G3's 280,640: its nodes, moves and references only 6,401
and its maps 117,480 (95%; ten levels on every empty cell at 1 + 4 bits). G3 needs 1,207 table moves, 6,330 leaves
and 3,359 exceptions — fewer, larger leaves reached earlier is the measured half of the gap; the other half, the
code."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_y1 import SPEC

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_weakc4.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_y1_weakc4_s_edge_is_a_small_tree_of_large_leaves():
    e = load_evidence(EVIDENCE / FILE)
    assert e["weakc4_commit"].startswith("51dc838") and (e["entries"], e["diagrams"]) == (654, 384)
    assert e["nodes"] == 1_431 and sum(e["moves_by_plies"].values()) == 251
    assert sum(e["leaves_by_plies"].values()) == 1_009 and sum(e["won_at_once_by_plies"].values()) == 171
    assert min(int(p) for p in e["leaves_by_plies"]) == 4
    assert (e["leaves_by_plies"]["4"], e["leaves_by_plies"]["6"]) == (18, 65)
    assert (e["maps"], e["mirrored_maps"]) == (745, 377)
    whole = e["readings"]["exact_all"]
    assert (whole["won"], whole["leaves"], whole["reasons"]) == (1_003, 1_009, {"won": 1_003, "undefined": 6})
    bits = whole["bits"]
    assert bits["total"] == 123_881 and round(bits["total"] / SPEC["g3_bits"], 2) == 0.44
    assert bits["nodes"] + bits["moves"] + bits["references"] == 6_401 and bits["maps"] == 117_480
    assert round(bits["maps"] / bits["total"], 2) == 0.95
