"""§3.6 J2 — finding registered AFTER the data: the whole game assembled by transposition group (harness.floor_j2)
from G3 and the builds of E1, O1, E3 and V1 (as tests/test_c49_v1_evidence.py assembles them) takes 197,650 bits, the
same 36 swaps J1's chooser found, whether it starts from J1's choice or from none. The builds the assembly leaves out
are not held back by the chooser but by the groups they sit in: 7 groups of ply-6 positions sharing ply-8 positions
hold an untaken build, and every one also holds positions with no build at all — 22 of them, 6 with a single winning
move (#40, #62, #76, #85, #102, #107), which no batch of every winning move at the positions with a choice builds. G3
pays 1,532 bits at the ply-8 positions #13 shares with #76, 1,664 at those #41 shares with #40, 2,090 in #99's group
and 2,448 in #73's: the next batch is chosen by group, single-move positions included."""
from __future__ import annotations

from collections import Counter

import pytest

from harness.floor_e3 import _root_move
from harness.floor_j1 import j1_choose, j1_piece
from harness.floor_j2 import j2_choose, j2_groups
from harness.floor_v1 import SPEC
from tests.test_c49_v1_evidence import EARLIER, EVIDENCE, FILES, LABELS, _assembly, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_j2_group_choice_adds_nothing_the_untaken_builds_wait_on_unbuilt_group_members():
    a = _assembly()
    runs = [r for name in EARLIER for r in _evidence(name)["roots"]] + _evidence()["roots"]
    index = {k: i for i, k in enumerate(sorted(a["below"]))}
    assert all(index[(tuple(r["board"]), r["to_move"])] == r["index"] for r in runs)
    candidates: dict = {}
    for r in runs:
        candidates.setdefault((tuple(r["board"]), r["to_move"]), []).append(
            ((r["index"], _root_move(r)), j1_piece(r, a["library"])))
    args = (a["opening"], a["below"], a["base"], candidates, a["map_cost"])
    j1, j1_size = j1_choose(*args)
    assert j1_size == 197_650 and len(j1) == 36
    assert j2_choose(*args, j1) == (j1, 197_650)
    assert j2_choose(*args, {}) == (j1, 197_650)
    reach = Counter(c for kids in a["below"].values() for c in kids)
    winning = {p["index"]: p["winning"] for p in _evidence(SPEC["scope"])["positions"]}
    held = {}
    for g in j2_groups(a["below"]):
        if any(k in candidates and k not in j1 for k in g) and len(g) > 1:
            shared = {c for k in g for c in a["below"][k] if reach[c] > 1 and c in a["base"]}
            held[tuple(sorted(index[k] for k in g))] = (sorted(index[k] for k in g if k not in candidates),
                                                        sum(a["base"][c][0] for c in shared))
    assert {g: bits for g, (_, bits) in held.items()} == {
        (13, 76): 1532, (27, 59, 61, 62, 63, 64, 65, 66, 73, 85): 2448, (37, 38, 43, 46, 49, 53, 136): 973,
        (39, 44, 47, 50, 54, 137): 0, (40, 41): 1664, (99, 102, 107): 2090, (127, 128): 921}
    unbuilt = sorted(i for members, _ in held.values() for i in members)
    assert len(unbuilt) == 22 and all(members for members, _ in held.values())
    assert [i for i in unbuilt if len(winning[i]) == 1] == [40, 62, 76, 85, 102, 107]
