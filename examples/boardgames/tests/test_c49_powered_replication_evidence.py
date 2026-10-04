"""§C.49 T17b — DESCRIPTIVE, registered after the data: which post-hoc readings of the calibration (h117) the powered
run reproduced on fresh seeds, and what a focused n-step replication needs. Read from
evidence/c49_P1_fixed_set.json.gz and evidence/c49_T17_fixed_set.json.gz. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from statistics import stdev

import pytest

from harness.evidence import load_evidence
from harness.fixed_set import calibration, pairs_needed
from harness.floor_c4_powered import SPEC, powered_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_P1_fixed_set.json.gz", "c49_T17_fixed_set.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _readings() -> tuple:
    post_hoc = calibration(load_evidence(EVIDENCE / FILES[0])["nets"])["paired"]
    powered = powered_report(load_evidence(EVIDENCE / FILES[1]), SPEC)["treatments"]
    return post_hoc, powered


def test_c49_T17b_only_n_step_kept_the_direction_the_calibration_showed_after_the_data():
    post_hoc, powered = _readings()
    before = {"n_step": "c49_T14/n_step", "tree_value": "c49_T14/tree_value", "backplay": "c49_T16/backplay"}
    assert all(post_hoc[k]["mean"] > 0 for k in before.values())
    assert {t: round(powered[t]["mean"], 3) for t in before} == {"n_step": 0.034, "tree_value": -0.004,
                                                                 "backplay": -0.023}


def test_c49_T17b_a_focused_n_step_replication_needs_9_seed_pairs_for_its_observed_gain():
    d = _readings()[1]["n_step"]["differences"]
    assert sum(1 for x in d if x > 0) == 5 and round(stdev(d), 3) == 0.04
    assert pairs_needed(stdev(d), 0.034) == 9
