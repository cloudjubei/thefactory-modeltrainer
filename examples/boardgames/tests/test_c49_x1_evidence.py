"""§3.6 X1 — PRE-REGISTERED claims on mirror sharing priced on G3's certified whole game, judged by harness.floor_x1
from evidence/c49_x1.json.gz (scripts/c4_mirror_sharing.py): storing a map that mirrors an earlier one as its index
saves at least a twentieth of G3's 280,640 bits, and playing a frontier position as its mirror image's strategy saves at
least a tenth."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_x1 import SPEC, x1_maps, x1_roots

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_x1.json.gz"
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def test_c49_x1_maps_stored_once_per_mirror_pair_save_a_twentieth():
    assert x1_maps(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_x1_the_map_saving_is_undecidable():
    assert x1_maps(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}


def test_c49_x1_frontier_positions_played_mirrored_save_a_tenth():
    assert x1_roots(_evidence(), SPEC)["verdict"] == "supported"


def test_c49_x1_the_root_saving_is_undecidable():
    assert x1_roots(_evidence(), SPEC)["verdict"] in {"inconclusive", "not_run"}
