"""§3.6 X2 — PRE-REGISTERED claims on reading the library on the mirrored board: the 51 builds the whole game takes
(180,199 bits; harness.floor_z1's game), each rebuilt at the same position with the same root move, with the builder
Z1 rebuilt them with and from the same 2,598 maps, every map also readable on the mirrored board — judged by
harness.floor_x2 from evidence/c49_x2.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_x2): 48 or more of the
51 complete, and with the rebuilds chosen by U1's search the whole game takes no more than 97/100 of 180,199."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g4 import SPEC as G4
from harness.floor_x2 import SPEC, x2_report, x2_size
from harness.floor_z1 import SPEC as Z1
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_z1_evidence import EARLIER, FILES as Z1_FILES, LABELS

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_x2.json.gz"
FILES = (FILE, *Z1_FILES)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=10)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return x2_size(_evidence(), (_evidence("c49_z1.json.gz"), Z1), earlier, (_evidence("c49_g4.json.gz"), G4),
                   _evidence("c49_g3.json.gz"), _assembly(), SPEC)


def test_c49_x2_48_of_the_51_rebuilds_complete():
    assert x2_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_x2_completion_is_undecidable():
    assert x2_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_x2_the_whole_game_with_the_mirror_takes_97_100_of_180_199_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_x2_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
