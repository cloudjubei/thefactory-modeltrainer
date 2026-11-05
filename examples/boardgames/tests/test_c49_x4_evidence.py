"""§3.6 X4 — PRE-REGISTERED claims on what the second reading itself is worth at ply 8: the 18 rebuilds that played a
mirrored leaf in X3, rebuilt at the same positions from the same list of 5,196 maps with the mirror off — judged by
harness.floor_x4 from evidence/c49_x4.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_x4): all 18 complete, and
with them in X3's place and U1's search run again the whole game takes 101/100 of 166,623 or more."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g4 import SPEC as G4
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC, x4_report, x4_size
from harness.floor_z1 import SPEC as Z1
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER, FILES as X3_FILES, LABELS

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_x4.json.gz"
FILES = (FILE, *X3_FILES)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=12)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return x4_size(_evidence(), (_evidence("c49_x3.json.gz"), X3), (_evidence("c49_x2.json.gz"), X2),
                   (_evidence("c49_z1.json.gz"), Z1), earlier, (_evidence("c49_g4.json.gz"), G4),
                   _evidence("c49_g3.json.gz"), _assembly(), SPEC)


def test_c49_x4_all_18_rebuilds_complete():
    assert x4_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_x4_completion_is_undecidable():
    assert x4_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_x4_with_the_mirror_off_the_whole_game_takes_101_100_of_166_623_or_more():
    assert _size()["verdict"] == "supported"


def test_c49_x4_the_mirror_s_worth_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
