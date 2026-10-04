"""§C.49 T14b — DESCRIPTIVE, registered after the data: what the two refuted value-signal fixes did to the value
head at the nets' errors. Read from evidence/c49_T14_readout.json.gz through
harness.floor_c4_value_signal.pilot_report. The register pins this file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_c4_value_signal import SPEC, pilot_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T14_readout.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _report():
    return pilot_report(load_evidence(EVIDENCE / FILE), SPEC)


def test_c49_T14b_search_value_targets_double_how_often_the_value_head_backs_the_net_s_errors():
    errors = _report()["at_errors"]
    assert errors["base"] == {"errors": 28, "value_backs": 10, "label_optimal": 5}
    assert errors["tree_value"] == {"errors": 33, "value_backs": 24, "label_optimal": 9}


def test_c49_T14b_n_step_targets_nearly_double_the_errors():
    errors = _report()["at_errors"]
    assert errors["n_step"] == {"errors": 52, "value_backs": 25, "label_optimal": 4}
    assert _report()["treatments"]["n_step"]["wins"] == 0
