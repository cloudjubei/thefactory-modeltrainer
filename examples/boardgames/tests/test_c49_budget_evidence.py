"""§C.49 T7 — the PRE-REGISTERED claims that T6's process reaches a raw-perfect net at the ~1K standardised-input
setups once the settle matches the oracle fit's optimisation budget (one claim per arm), judged by
harness.floor_budget.budget_report from the three arm files. Each claim's proof passes only on SUPPORTED; its
undecidable proof passes on INCONCLUSIVE or NOT_RUN. The register pins this file, harness/floor_budget.py and
harness/arm_judge.py."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_budget import SPEC, budget_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
ARM_FILES = {name: f"c49_T7_{name}.json.gz" for name in SPEC["arms"]}
FILES = tuple(ARM_FILES.values())
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return budget_report({name: load_evidence(EVIDENCE / f) for name, f in ARM_FILES.items()})


@pytest.mark.parametrize("arm", sorted(SPEC["arms"]))
def test_c49_T7_with_the_oracle_budget_the_process_gives_a_raw_perfect_net_on_8_of_10_seeds(arm):
    assert _report()[arm]["verdict"] == "supported"


@pytest.mark.parametrize("arm", sorted(SPEC["arms"]))
def test_c49_T7_arm_is_undecidable(arm):
    assert _report()[arm]["verdict"] in {"inconclusive", "not_run"}
