"""§3.7 BF1 — PRE-REGISTERED claims on best-first proving with maps, judged by harness.floor_bf1 from
evidence/c49_bf1.json.gz (scripts/c4_best_first.py): on a fresh sample of T8's leaf positions, at the same work,
best-first proving with maps proves at least twice as many as depth-first proving with maps and as best-first proving
with no maps; and every proof any arm gives holds. Each stored proof is rebuilt here and played by harness.composed's
check, independently of the run's own word."""
from __future__ import annotations

from functools import lru_cache

import pytest

from games.connect4 import C4State, Connect4
from harness.composed import check
from harness.floor_bf1 import SPEC, bf1_report
from harness.steady_state import Facts
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

FILES = (SPEC["file"], SPEC["source"])
CHECK_CAP = 10_000_000
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _strategy(proof: dict) -> dict:
    return {"nodes": {(tuple(board), to_move): node for board, to_move, node in proof["nodes"]},
            "maps": [{c: v for c, v in m} for m in proof["maps"]]}


def _check(row: dict, arm: str) -> bool:
    facts = Facts(Connect4(), **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})
    state = C4State(tuple(row["board"]), row["to_move"], None, False)
    proof = row["arms"][arm].get("proof")
    return proof is not None and check(facts, state, _strategy(proof), SPEC["rule"]["n_levels"], CHECK_CAP)["won"]


@lru_cache(maxsize=1)
def _report() -> dict:
    return bf1_report(_evidence(SPEC["file"]), _evidence(SPEC["source"]), SPEC, _check)


def test_c49_bf1_best_first_with_maps_proves_twice_as_many_as_depth_first():
    assert _report()["best_vs_depth"] == "supported"


def test_c49_bf1_best_vs_depth_is_undecidable():
    assert _report()["best_vs_depth"] in {"inconclusive", "not_run"}


def test_c49_bf1_best_first_with_maps_proves_twice_as_many_as_with_no_maps():
    assert _report()["best_vs_nomaps"] == "supported"


def test_c49_bf1_best_vs_nomaps_is_undecidable():
    assert _report()["best_vs_nomaps"] in {"inconclusive", "not_run"}


def test_c49_bf1_every_proof_holds():
    assert _report()["sound"] == "supported"


def test_c49_bf1_soundness_is_undecidable():
    assert _report()["sound"] in {"inconclusive", "not_run"}
