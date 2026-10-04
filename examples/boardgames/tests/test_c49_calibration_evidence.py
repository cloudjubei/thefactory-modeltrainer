"""§C.49 P1 — DESCRIPTIVE, registered after the data: how much Connect-4 nets of one process vary when every net is
scored on ONE FIXED SET of exactly valued positions (the label cache's White positions at plies 0-8), read from
evidence/c49_P1_fixed_set.json.gz (scripts/c4_fixed_set_readout.py over the 20 T14/T16 nets) through
harness.fixed_set. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.fixed_set import calibration, pairs_needed

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_P1_fixed_set.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _calibration() -> dict:
    e = load_evidence(EVIDENCE / FILE)
    assert e["positions"] == 5142 and len(e["nets"]) == 20
    return calibration(e["nets"])


def _pooled_paired_sd(c: dict) -> float:
    sds = [v["sd"] for v in c["paired"].values()]
    return (sum(s * s for s in sds) / len(sds)) ** 0.5


def test_c49_P1_nets_of_one_arm_spread_3_5_points_and_seed_matched_differences_3_1():
    c = _calibration()
    assert round(c["within_arm_sd"], 3) == 0.035 and round(_pooled_paired_sd(c), 3) == 0.031


def test_c49_P1_a_paired_pilot_needs_7_pairs_for_3_points_and_16_for_2():
    sd = _pooled_paired_sd(_calibration())
    assert (pairs_needed(sd, 0.05), pairs_needed(sd, 0.03), pairs_needed(sd, 0.02)) == (3, 7, 16)


def test_c49_P1_post_hoc_all_three_refuted_treatments_read_slightly_above_base_on_the_fixed_set():
    paired = _calibration()["paired"]
    means = {k: round(v["mean"], 3) for k, v in paired.items()}
    assert means == {"c49_T14/n_step": 0.021, "c49_T14/tree_value": 0.019, "c49_T16/backplay": 0.017}
    assert all(sum(1 for d in v["differences"] if d > 0) == 3 for v in paired.values())
