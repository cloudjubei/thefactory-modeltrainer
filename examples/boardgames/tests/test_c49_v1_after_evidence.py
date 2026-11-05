"""§3.6 V1 — finding registered AFTER the data, from evidence/c49_v1.json.gz with the whole game J1's chooser
assembles (harness.floor_j1, as tests/test_c49_v1_evidence.py builds it): all 63 builds complete (115.5 hours, none
over 5.1), and the whole game takes 197,650 bits — 36 ply-6 positions take a build, 8 of them V1's, 476 of G3's
strategies and 847 library maps remain. Of V1's 22 positions, 5 take no build, and at 4 of them (#13, #41, #73, #99)
every ply-8 position below is also reached from another ply-6 position that keeps G3's piece, so G3's strategies there
stay and a build saves at most the 4-bit move — #13's move 3 takes 66 bits against G3's 1,536 below it and is not
taken. Such a build pays only if every ply-6 position sharing those ply-8 positions swaps with it, and the chooser
tries no more than pairs: 197 of the 820 ply-8 positions are reached from more than one ply-6 position, joining the
148 into groups of up to 10."""
from __future__ import annotations

from collections import Counter

import pytest

from harness.floor_e3 import _root_move
from harness.floor_g1 import _coded
from harness.floor_j1 import j1_bits, j1_choose, j1_piece
from tests.test_c49_v1_evidence import EARLIER, EVIDENCE, FILES, LABELS, _assembly, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _groups(below: dict) -> list:
    parent = {k: k for k in below}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    owner: dict = {}
    for k, kids in below.items():
        for c in kids:
            if c in owner:
                parent[find(k)] = find(owner[c])
            else:
                owner[c] = k
    return sorted(Counter(find(k) for k in below).values())


def test_c49_v1_builds_whose_ply_8_positions_other_positions_share_are_not_taken():
    v = _evidence()
    assert all(r["complete"] for r in v["roots"]) and len(v["roots"]) == 63
    assert round(sum(r["build_seconds"] for r in v["roots"]) / 3600, 1) == 115.5
    assert max(r["build_seconds"] for r in v["roots"]) < 5.1 * 3600
    a = _assembly()
    candidates: dict = {}
    for n, rows in enumerate([[r for name in EARLIER for r in _evidence(name)["roots"]], v["roots"]]):
        for r in rows:
            candidates.setdefault((tuple(r["board"]), r["to_move"]), []).append(
                ((n, r["index"], _root_move(r)), j1_piece(r, a["library"])))
    chosen, bits = j1_choose(a["opening"], a["below"], a["base"], candidates, a["map_cost"])
    built = {k: dict(candidates[k])[label] for k, label in chosen.items()}
    assert j1_bits(a["opening"], a["below"], a["base"], built, a["map_cost"]) == {
        "bits": 197_650, "kept": 112, "trivial": 137, "strategies": 476, "library_maps": 847}
    assert len(chosen) == 36 and sorted([i, m] for n, i, m in chosen.values() if n == 1) == [
        [1, 2], [3, 1], [4, 5], [71, 2], [72, 5], [80, 2], [118, 3], [143, 1]]
    index = {(tuple(r["board"]), r["to_move"]): r["index"] for r in v["roots"]}
    kept = [k for k in a["below"] if k not in built]
    untaken = sorted(i for k, i in index.items() if k not in built)
    shared = sorted(i for k, i in index.items() if k not in built
                    and all(any(c in a["below"][q] for q in kept if q != k) for c in a["below"][k]))
    assert untaken == [13, 41, 73, 99, 141] and shared == [13, 41, 73, 99]
    assert {r["root_move"]: _coded(r) for r in v["roots"] if r["index"] == 13}[3] == 66
    reach = Counter(c for kids in a["below"].values() for c in kids)
    assert (sum(1 for n in reach.values() if n > 1), len(reach)) == (197, 820)
    assert len(a["below"]) == 148 and max(_groups(a["below"])) == 10
