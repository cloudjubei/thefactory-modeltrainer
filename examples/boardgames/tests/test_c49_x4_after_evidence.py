"""§3.6 X4 — finding registered AFTER the data, from evidence/c49_x4.json.gz: what the second reading is worth at ply 8.
With the 18 mirror-off rebuilds in X3's place and U1's search run again the whole game takes 169,200 bits, 2,577 more
than X3's 166,623 (1.0155 of it); the search stops at 68 taken from all and at 55 taken from none (169,493). The 18 take
9,279 bits against 6,677 with the mirror on; 15 are larger, 3 smaller, and the game takes 8 of them against X3's 13. No
build reached the 2-hour cap, and the 18 took 8.0 build-hours against X3's 10.0."""
from __future__ import annotations

import pytest

import harness.floor_x3 as floor_x3
from harness.floor_g4 import SPEC as G4, _key
from harness.floor_j1 import _coded
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC, x4_size
from harness.floor_z1 import SPEC as Z1
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER
from tests.test_c49_x4_evidence import EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_x4_with_the_mirror_off_the_whole_game_takes_2_577_bits_more(monkeypatch):
    found, real = [], floor_x3.u1_choose
    monkeypatch.setattr(floor_x3, "u1_choose", lambda size, plays, start: found.append(real(size, plays, start))
                        or found[-1])
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    r = x4_size(_evidence(), (_evidence("c49_x3.json.gz"), X3), (_evidence("c49_x2.json.gz"), X2),
                (_evidence("c49_z1.json.gz"), Z1), earlier, (_evidence("c49_g4.json.gz"), G4),
                _evidence("c49_g3.json.gz"), _assembly(), SPEC)
    assert r == {"verdict": "supported", "bits": 169_200, "control_bits": 166_623, "ratio": 169_200 / 166_623,
                 "rebuilt": 68, "unfinished": False}
    assert [(len(held), bits) for held, bits in found] == [(54, 167_046), (71, 166_623), (55, 169_493), (68, 169_200)]
    off = _evidence()["roots"]
    on = {x["index"]: x for x in _evidence("c49_x3.json.gz")["roots"]}
    change = [_coded(x) - _coded(on[x["index"]]) for x in off]
    assert (sum(_coded(x) for x in off), sum(_coded(on[x["index"]]) for x in off)) == (9_279, 6_677)
    assert (sum(c > 0 for c in change), sum(c < 0 for c in change)) == (15, 3)
    keys = {_key(x) for x in off}
    assert (len(keys & found[3][0]), len(keys & found[1][0])) == (8, 13)
    assert max(x["build_seconds"] for x in off) < SPEC["builder"]["seconds"]
    assert round(sum(x["build_seconds"] for x in off) / 3600, 1) == 8.0
    assert round(sum(on[x["index"]]["build_seconds"] for x in off) / 3600, 1) == 10.0
