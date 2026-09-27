"""§C.49 T6 — the PRE-REGISTERED claims that T5's working tic-tac-toe process gives a raw-perfect net at the oracle
frontier setups (one claim per arm), judged by harness.floor_small.small_report from the five arm files. Each claim's
proof passes only on SUPPORTED; its undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file
and harness/floor_small.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_small import T6_ARMS, small_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
ARM_FILES = {name: f"c49_T6_{name}.json.gz" for name in T6_ARMS}
FILES = tuple(ARM_FILES.values())
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return small_report({name: load_evidence(EVIDENCE / f) for name, f in ARM_FILES.items()})


@pytest.mark.parametrize("arm", sorted(T6_ARMS))
def test_c49_T6_the_process_gives_a_raw_perfect_net_at_the_frontier_setup_on_8_of_10_seeds(arm):
    assert _report()[arm]["verdict"] == "supported"


@pytest.mark.parametrize("arm", sorted(T6_ARMS))
def test_c49_T6_arm_is_undecidable(arm):
    assert _report()[arm]["verdict"] in {"inconclusive", "not_run"}
