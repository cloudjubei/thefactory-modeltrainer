"""§3.6 L4 — findings registered AFTER the data, from evidence/c49_l4.json.gz (L2's rule preferring forcing moves) and
evidence/c49_l4_control.json.gz (the same without), on F1's 16 heaviest and 16 spread positions, with
evidence/c49_f1.json.gz. Preferring forcing moves starves the leaf search: its searches find a map 766 times in
6,464 (12%) against the control's 1,213 in 2,753 (44%), so its 28 unfinished builds each carry 171-201 table moves
when the 2-hour cap ends them (the control at most 70), and on the four positions both complete it is never smaller
(4,480 bits against 658, 2,674 against 102, 117 against 90, 39 against 39). The control — the same rule, no library,
another day — builds the 32 in 130,061 bits against F1's 144,144 with its growing library: smaller on 20 positions,
larger on 12, the heaviest 16 still 93% of the bits."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l4 import SPEC

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l4.json.gz"
FILES = (FILE, "c49_l4_control.json.gz", "c49_f1.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=3)
def _roots(name: str) -> dict:
    return {r["index"]: r for r in load_evidence(EVIDENCE / name)["roots"]}


def test_c49_l4_preferring_forcing_moves_starves_the_leaf_search():
    forcing, control = _roots(FILE), _roots("c49_l4_control.json.gz")
    assert (sum(r["searches"]["tried"] for r in forcing.values()),
            sum(r["searches"]["found"] for r in forcing.values())) == (6_464, 766)
    assert (sum(r["searches"]["tried"] for r in control.values()),
            sum(r["searches"]["found"] for r in control.values())) == (2_753, 1_213)
    unfinished = [r["bits"]["moves"] for r in forcing.values() if not r["complete"]]
    assert len(unfinished) == 28 and (min(unfinished), max(unfinished)) == (171, 201)
    assert max(r["bits"]["moves"] for r in control.values()) == 70
    both = sorted((i, r["bits"]["nodes"], control[i]["bits"]["nodes"]) for i, r in forcing.items() if r["complete"])
    assert both == [(9, 39, 39), (15, 4_480, 658), (353, 117, 90), (568, 2_674, 102)]


def test_c49_l4_the_control_rebuilds_f1_s_positions_a_tenth_smaller_without_a_library():
    control, f1 = _roots("c49_l4_control.json.gz"), _roots("c49_f1.json.gz")
    assert all(r["complete"] for r in control.values()) and sum(r["library_leaves"] for r in control.values()) == 0
    assert sum(r["bits"]["nodes"] for r in control.values()) == 130_061
    assert sum(f1[i]["bits"]["nodes"] for i in SPEC["indices"]) == 144_144
    assert sum(1 for i, r in control.items() if r["bits"]["nodes"] < f1[i]["bits"]["nodes"]) == 20
    assert sum(1 for i, r in control.items() if r["bits"]["nodes"] > f1[i]["bits"]["nodes"]) == 12
    assert sum(control[i]["bits"]["nodes"] for i in SPEC["indices"][:16]) == 121_313
