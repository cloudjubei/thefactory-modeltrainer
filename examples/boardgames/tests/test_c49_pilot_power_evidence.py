"""§C.49 T16b — DESCRIPTIVE, registered after the data: how noisy the 4-seed Connect-4 pilot measure is. The same
unchanged base process, read the same way, on two sets of four seeds (T14: 401-404, T16: 411-414). Read from
evidence/c49_T14_readout.json.gz and evidence/c49_T16_readout.json.gz through
harness.floor_c4_value_signal.pilot_report. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_c4_backplay import SPEC as T16_SPEC
from harness.floor_c4_value_signal import SPEC as T14_SPEC, pilot_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_T14_readout.json.gz", "c49_T16_readout.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _bases() -> tuple:
    t14 = pilot_report(load_evidence(EVIDENCE / FILES[0]), T14_SPEC)
    t16 = pilot_report(load_evidence(EVIDENCE / FILES[1]), T16_SPEC)
    return t14["nets"]["base"], t16["nets"]["base"]


def _pooled(nets: dict) -> float:
    return sum(n["optimal"] for n in nets.values()) / sum(n["positions"] for n in nets.values())


def test_c49_T16b_the_unchanged_process_scores_13_points_apart_on_two_sets_of_four_seeds():
    t14, t16 = _bases()
    assert round(_pooled(t14), 3) == 0.859 and round(_pooled(t16), 3) == 0.731


def test_c49_T16b_single_base_nets_range_from_62_to_94_percent():
    shares = [n["optimal"] / n["positions"] for nets in _bases() for n in nets.values()]
    assert len(shares) == 8 and round(min(shares), 2) == 0.62 and round(max(shares), 2) == 0.94
