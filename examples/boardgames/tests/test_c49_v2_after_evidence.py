"""§3.6 V2 — finding registered AFTER the data, from evidence/c49_v2.json.gz with the whole game assembled as
harness.floor_v2 assembles it (J1's chooser, then by group): all 67 builds complete (107.5 hours, each complete before
the 6-hour cap; #87's move 6, complete at 4.2 hours, improved until its deadline and stopped 2.9 seconds after), and
the whole game takes 192,591 bits — 0.974 of 197,650, 871 bits over the 97/100 bar. 16 of V2's builds are taken, and
the group of #91 to #114 gives 1,920 of the 5,059 bits saved (4,208 to 2,288). The free builds at the heavy
single-move positions save little: #2 4,418 bits and #68 4,350 against G3's 4,886 and 4,894 below them, and #96's
free build takes 4,842 bits against G3's 4,304, so is not taken. Every winning move is now built in all but one of
the 11 groups on which the game still pays at least 3,000 bits (56,552 bits, 29% of it): a batch by group at 3,000
bits holds only #116's free build, and at 2,000 bits 51 builds in 8 groups. The parts: opening 845, library 72,637
(834 maps), 409 of G3's strategies 61,427, builds at 51 ply-6 positions 57,682 (V2's 17,282)."""
from __future__ import annotations

import pytest

from harness.floor_j1 import j1_bits, j1_piece
from harness.floor_j2 import j2_groups
from harness.floor_v2 import SPEC, v2_assemble, v2_selected
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_v2_evidence import EARLIER, EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_v2_saves_2_6_percent_and_the_heaviest_groups_have_every_winning_move_built():
    v = _evidence()["roots"]
    assert len(v) == 67 and all(r["complete"] for r in v)
    assert round(sum(r["build_seconds"] for r in v) / 3600, 1) == 107.5
    assert max(r["improved"]["complete_seconds"] for r in v) <= 6 * 3600
    assert max(r["build_seconds"] for r in v) < 6 * 3600 + 5
    a, rows = _assembly(), [r for name in EARLIER for r in _evidence(name)["roots"]]
    lib, index = a["library"], {k: i for i, k in enumerate(sorted(a["below"]))}
    before, control = v2_assemble(rows, a)
    pieces = rows + v
    chosen, bits = v2_assemble(pieces, a)
    assert (control, bits) == (197_650, 192_591)
    assert len([n for n in chosen.values() if n >= len(rows)]) == 16
    built = {k: j1_piece(pieces[n], lib) for k, n in chosen.items()}
    assert j1_bits(a["opening"], a["below"], a["base"], built, a["map_cost"]) == {
        "bits": 192_591, "kept": 97, "trivial": 123, "strategies": 409, "library_maps": 834}
    kids = {c for k in a["below"] if k not in chosen for c in a["below"][k] if c in a["base"]}
    maps = set().union(*(p[1] for p in built.values()), *(a["base"][c][1] for c in kids))
    assert sum(p[0] for p in built.values()) == 57_682 and len(built) == 51
    assert sum(j1_piece(pieces[n], lib)[0] for n in chosen.values() if n >= len(rows)) == 17_282
    assert sum(a["base"][c][0] for c in kids) == 61_427
    assert sum(a["map_cost"][m] for m in maps) == 72_637

    def paid(group, choice, source):
        kept = [k for k in group if k not in choice]
        below = {c for k in kept for c in a["below"][k]}
        return (4 * len(kept) + sum(a["base"][c][0] if c in a["base"] else 1 for c in below)
                + sum(j1_piece(source[choice[k]], lib)[0] for k in group if k in choice))

    groups = {tuple(index[k] for k in g): g for g in j2_groups(a["below"])}
    assert [paid(groups[(91, 98, 104, 105, 110, 112, 114)], c, s) for c, s in ((before, rows), (chosen, pieces))] == [
        4208, 2288]
    single = {i: (paid(groups[(i,)], before, rows), paid(groups[(i,)], chosen, pieces)) for i in (2, 68, 96)}
    assert single == {2: (4886, 4418), 68: (4894, 4350), 96: (4304, 4304)}
    assert [j1_piece(r, lib)[0] for r in v if r["index"] == 96] == [4842]
    heavy = sorted(((paid(g, chosen, pieces), i) for i, g in groups.items()), reverse=True)
    assert len([p for p, _ in heavy if p >= 3000]) == 11 and sum(p for p, _ in heavy if p >= 3000) == 56_552
    scope = _evidence(SPEC["scope"])
    assert v2_selected(scope, pieces, a, {**SPEC, "min_group_bits": 3000}) == [[116, None]]
    at_2000 = v2_selected(scope, pieces, a, {**SPEC, "min_group_bits": 2000})
    assert len(at_2000) == 51
    assert len({g for p, g in heavy if p >= 2000 and any(i in g for i, _ in at_2000)}) == 8
