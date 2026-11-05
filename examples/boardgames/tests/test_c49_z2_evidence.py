"""§3.6 Z2 — PRE-REGISTERED claims on the ply-6 builds of U2's game rebuilt from every map the game holds: judged by
harness.floor_z2 from evidence/c49_z2.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_z2): at least 47 of the
50 complete, and with every rebuild an option for U2's search the whole game takes 99/100 of 165,023 or less."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_g4 import SPEC as G4
from harness.floor_u2 import SPEC as U2
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC as X4
from harness.floor_z1 import SPEC as Z1
from harness.floor_z2 import SPEC, z2_report, z2_size
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER
from tests.test_c49_x4_evidence import FILES as X4_FILES, LABELS

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_z2.json.gz"
FILES = (FILE, *X4_FILES)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=13)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return z2_size(_evidence(), (_evidence("c49_x4.json.gz"), X4), (_evidence("c49_x3.json.gz"), X3),
                   (_evidence("c49_x2.json.gz"), X2), (_evidence("c49_z1.json.gz"), Z1), earlier,
                   (_evidence("c49_g4.json.gz"), G4), _evidence("c49_g3.json.gz"), _assembly(), U2, SPEC)


def test_c49_z2_47_of_the_50_rebuilds_complete():
    assert z2_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_z2_completion_is_undecidable():
    assert z2_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_z2_the_whole_game_takes_99_100_of_165_023_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_z2_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
