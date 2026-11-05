"""§3.6 Z1 — PRE-REGISTERED claims on rebuilding the assembled whole game's ply-6 builds: the 51 builds the whole game
takes (184,328 bits; harness.floor_u1's game, its ply-8 strategies G3's and the 66 rebuilds of G4 that U1 keeps), each
rebuilt at the same position with the same root move, with the builder they were built with, from every map that game
holds — G4's library, then the own maps of those 66 — judged by harness.floor_z1 from evidence/c49_z1.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_z1): 48 or more of the 51 complete, and with the rebuilds chosen by
U1's search the whole game takes no more than 97/100 of 184,328."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_e1 import SPEC as E1
from harness.floor_e3 import SPEC as E3
from harness.floor_g4 import SPEC as G4
from harness.floor_o1 import SPEC as O1
from harness.floor_v1 import SPEC as V1
from harness.floor_v2 import SPEC as V2
from harness.floor_z1 import SPEC, z1_report, z1_size
from tests.test_c49_v1_evidence import _assembly

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_z1.json.gz"
EARLIER = dict(zip(SPEC["earlier"], (E1, O1, E3, V1, V2)))
FILES = (FILE, *EARLIER, "c49_g4.json.gz", "c49_g3.json.gz", "c49_g2.json.gz", "c49_f2.json.gz")
LABELS = ROOT / "books" / "c4_labels.json.gz"
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=9)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return z1_size(_evidence(), earlier, (_evidence("c49_g4.json.gz"), G4), _evidence("c49_g3.json.gz"), _assembly(),
                   SPEC)


def test_c49_z1_48_of_the_51_rebuilds_complete():
    assert z1_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_z1_completion_is_undecidable():
    assert z1_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_z1_the_whole_game_with_the_rebuilds_takes_97_100_of_184_328_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_z1_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
