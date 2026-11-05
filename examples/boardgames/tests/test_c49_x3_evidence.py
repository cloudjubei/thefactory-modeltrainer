"""§3.6 X3 — PRE-REGISTERED claims on reading the library on the mirrored board at ply 8: the 409 ply-8 strategies the
whole game takes (175,425 bits; harness.floor_x2's game), each rebuilt at the same position with the builder G4 rebuilt
them with, from the 2,598 maps X2 read both ways — judged by harness.floor_x3 from evidence/c49_x3.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_x3): all 409 complete, and with the rebuilds chosen by U1's search the
whole game takes no more than 97/100 of 175,425."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g4 import SPEC as G4
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC, x3_report, x3_size
from harness.floor_z1 import SPEC as Z1
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x2_evidence import FILES as X2_FILES
from tests.test_c49_z1_evidence import EARLIER, LABELS

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_x3.json.gz"
FILES = (FILE, *X2_FILES)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=11)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return x3_size(_evidence(), (_evidence("c49_x2.json.gz"), X2), (_evidence("c49_z1.json.gz"), Z1), earlier,
                   (_evidence("c49_g4.json.gz"), G4), _evidence("c49_g3.json.gz"), _assembly(), SPEC)


def test_c49_x3_all_409_rebuilds_complete():
    assert x3_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_x3_completion_is_undecidable():
    assert x3_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_x3_the_whole_game_with_the_mirror_takes_97_100_of_175_425_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_x3_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
