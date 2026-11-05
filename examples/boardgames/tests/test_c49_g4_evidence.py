"""§3.6 G4 — PRE-REGISTERED claims on the assembled whole game's library pass: the 409 ply-8 positions the whole game
assembled from G3 and the builds of E1, O1, E3, V1 and V2 (192,591 bits; harness.floor_v2's assembly, as
tests/test_c49_v1_evidence.py builds it) still takes from G3, each rebuilt with G3's builder from every map that game
holds — G3's library, then those G3 strategies' own maps, then the own maps of the 51 builds it takes — judged by
harness.floor_g4 from evidence/c49_g4.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_g4): all 409 complete,
and with them in place (a rebuild that alone grows the game going back to G3's, the ply-6 builds chosen again) the
whole game takes no more than 97/100 of 192,591."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_e1 import SPEC as E1
from harness.floor_e3 import SPEC as E3
from harness.floor_g4 import SPEC, g4_report, g4_size
from harness.floor_o1 import SPEC as O1
from harness.floor_v1 import SPEC as V1
from harness.floor_v2 import SPEC as V2
from tests.test_c49_v1_evidence import _assembly

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_g4.json.gz"
EARLIER = dict(zip(SPEC["earlier"], (E1, O1, E3, V1, V2)))
FILES = (FILE, *EARLIER, "c49_g3.json.gz", "c49_g2.json.gz", "c49_f2.json.gz")
LABELS = ROOT / "books" / "c4_labels.json.gz"
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=8)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return g4_size(_evidence(), earlier, _evidence("c49_g3.json.gz"), _assembly(), SPEC)


def test_c49_g4_all_409_rebuilds_complete():
    assert g4_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_g4_completion_is_undecidable():
    assert g4_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_g4_the_whole_game_with_the_pass_takes_97_100_of_192_591_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_g4_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
