"""§3.6 E3 — PRE-REGISTERED claims on every other winning move at E1's ply-6 positions: the 13 moves the scope
(evidence/c49_o1_scope.json.gz) lists as winning there that E1's roots did not play, each built from ply 6 with that
move at the root and otherwise as E1 built its 8, judged by harness.floor_e3 from evidence/c49_e3.json.gz
(scripts/c4_strategy_s2.py --spec harness.floor_e3) against E1 (evidence/c49_e1.json.gz) and G3
(evidence/c49_g3.json.gz, its library from c49_g2 and c49_f2): at least 12 of the 13 complete within 6 hours, and the
cheapest complete build at each of the 7 positions takes no more than nine tenths of E1's 28,712 bits for them."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_e3 import SPEC, e3_report, e3_size
from harness.floor_g3 import g3_library

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
FILE = "c49_e3.json.gz"
FILES = (FILE, "c49_e1.json.gz", SPEC["scope"], "c49_g3.json.gz", "c49_g2.json.gz", "c49_f2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=6)
def _evidence(name: str = FILE) -> dict:
    return load_evidence(EVIDENCE / name)


@lru_cache(maxsize=1)
def _size() -> dict:
    library = g3_library(_evidence("c49_g2.json.gz"), _evidence("c49_f2.json.gz"))
    return e3_size(_evidence(), _evidence("c49_e1.json.gz"), _evidence("c49_g3.json.gz"), library,
                   _evidence(SPEC["scope"]), SPEC)


def test_c49_e3_12_of_13_other_winning_moves_complete_within_6_hours():
    assert e3_report(_evidence(), SPEC)["complete"]["verdict"] == "supported"


def test_c49_e3_completion_is_undecidable():
    assert e3_report(_evidence(), SPEC)["complete"]["verdict"] in {"inconclusive", "not_run"}


def test_c49_e3_the_cheapest_winning_move_takes_a_tenth_less_than_the_builder_s_pick():
    assert _size()["verdict"] == "supported"


def test_c49_e3_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
