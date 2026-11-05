"""§3.6 E3 — finding registered AFTER the data, from evidence/c49_e3.json.gz against E1 (evidence/c49_e1.json.gz) and
the scope (evidence/c49_o1_scope.json.gz): the builder's free pick is the cheapest move only where it differs from the
opening's. At #115, #130 and #145 (the pick moved off the opening's lowest column) no other winning move beats E1's
build; at the other four another move does — #18 4 (4,392 against 6,231), #69 1 (3,706 against 5,621), #79 4 (2,340
against 6,036), #146 4 (3,061 against 6,373) — two where the pick was a tie on covered replies broken by the lowest
column (#18, #146) and two where the move with the most covered replies lost to one with fewer (#69, #79). Moves at one
position differ up to 13-fold (#145: 373 to 4,970); 35.1 hours of builds."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_e3 import SPEC, _root_move
from harness.floor_g1 import _coded

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILES = ("c49_e3.json.gz", "c49_e1.json.gz", SPEC["scope"])
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_e3_the_free_pick_is_cheapest_only_where_it_differs_from_the_opening_s():
    e3, e1, scope = (load_evidence(EVIDENCE / f) for f in FILES)
    positions = {p["index"]: p for p in scope["positions"]}
    bits = {}
    for r in e1["roots"] + e3["roots"]:
        if r["index"] in SPEC["indices"]:
            bits.setdefault(r["index"], {})[_root_move(r)] = _coded(r)
    assert bits == {18: {2: 6231, 4: 4392}, 69: {0: 5621, 1: 3706, 6: 16285},
                    79: {0: 6036, 1: 6293, 4: 2340, 6: 8446}, 115: {4: 1993, 2: 5908}, 130: {5: 474, 2: 6265},
                    145: {2: 373, 0: 4970, 1: 1278, 4: 750}, 146: {1: 6373, 4: 3061, 5: 5168}}
    picks = {i: positions[i]["library_pick"] for i in bits}
    moved = sorted(i for i in bits if picks[i] != positions[i]["opening"])
    assert moved == [115, 130, 145]
    assert sorted(i for i, b in bits.items() if min(b, key=b.get) == picks[i]) == moved
    covered = {i: {a: c for c, _, a in positions[i]["scored"]} for i in bits}
    tied = sorted(i for i in bits if i not in moved and list(covered[i].values()).count(covered[i][picks[i]]) > 1)
    assert tied == [18, 146]
    assert all(covered[i][min(bits[i], key=bits[i].get)] < covered[i][picks[i]] for i in (69, 79))
    assert round(sum(r["build_seconds"] for r in e3["roots"]) / 3600, 1) == 35.1
