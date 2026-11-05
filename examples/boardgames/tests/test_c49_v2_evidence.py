"""§3.6 V2 — PRE-REGISTERED claims on every winning move by transposition group: in the 9 groups of ply-6 positions
on which the whole game assembled from E1, O1, E3 and V1 (J1's chooser, then by group) still pays at least 4,000
bits, the 67 builds no earlier root covers — each winning move at a member with more than one, a free build at a
member with one — each built from ply 6 and otherwise as E1 built its 8, judged by harness.floor_v2 from
evidence/c49_v2.json.gz (scripts/c4_strategy_s2.py --spec harness.floor_v2) against that whole game (G3,
evidence/c49_g3.json.gz with its library from c49_g2 and c49_f2, the ply-6 opening and the ply-8 opening's moves from
books/c4_labels.json.gz, as tests/test_c49_v1_evidence.py assembles them): at least 62 of the 67 complete within 6
hours, and with them added the whole game takes no more than 97/100 of its 197,650."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_e1 import SPEC as E1
from harness.floor_e3 import SPEC as E3
from harness.floor_o1 import SPEC as O1
from harness.floor_v1 import SPEC as V1
from harness.floor_v2 import SPEC, v2_report, v2_size
from tests.test_c49_v1_evidence import _assembly

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_v2.json.gz"
EARLIER = dict(zip(SPEC["earlier"], (E1, O1, E3, V1)))
FILES = (FILE, *EARLIER, SPEC["scope"], "c49_g3.json.gz", "c49_g2.json.gz", "c49_f2.json.gz")
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
    return v2_size(_evidence(), earlier, _evidence(SPEC["scope"]), _assembly(), SPEC)


def test_c49_v2_62_of_67_builds_by_group_complete_within_6_hours():
    assert v2_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_v2_completion_is_undecidable():
    assert v2_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_v2_the_whole_game_with_the_batch_takes_97_100_of_197_650_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_v2_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
