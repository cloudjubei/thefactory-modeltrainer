"""§C.49 H3 — findings registered AFTER the data, from evidence/c49_H3_readout.json.gz. The pre-registered judge
(harness.floor_h3.h3_report) did not run: the H3 nets were trained with 6 relabel workers, the registered recipe
(H2's table arm) says 4, so h134 and h135 read INCONCLUSIVE by protocol. These proofs record that slip, what the
registered tests give when the run's own worker count is substituted (relabelling across workers trains bit-identically
— tests/test_neural.py::test_c49_relabelling_across_worker_processes_trains_bit_identically), and the size of every
certified-through-9 hybrid against a table alone over the same tree (harness.description_length)."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import pytest

from harness.description_length import hybrid_bits
from harness.evidence import load_evidence
from harness.floor_h3 import SPEC, h3_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_H3_readout.json.gz"
FILES = (FILE,)
ACTIONS = 7
BITS_PER_PARAM = 32
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _readout() -> dict:
    return load_evidence(EVIDENCE / FILE)


def _as_run() -> dict:
    spec = copy.deepcopy(SPEC)
    spec["arms"]["h3"]["relabel_workers"] = _readout()["runs"]["h3"]["config"]["relabel_workers"]
    return h3_report(_readout(), spec)


def test_c49_H3_was_launched_with_6_relabel_workers_not_the_registered_4_so_the_judge_did_not_run():
    e = _readout()
    assert e["runs"]["h3"]["config"]["relabel_workers"] == 6 and SPEC["arms"]["h3"]["relabel_workers"] == 4
    r = h3_report(e, SPEC)
    assert r["integrity"] == ["h3: the nets were not trained on the registered recipe"]
    assert r["s8"] == {"verdict": "not_run"} and r["curve"] == {"verdict": "not_run"}


def test_c49_H3_with_the_run_s_own_worker_count_both_registered_tests_are_still_inconclusive():
    r = _as_run()
    assert r["integrity"] == []
    assert r["s8"]["verdict"] == "inconclusive" and 0.011 < r["s8"]["mean"] < 0.013 and r["s8"]["p"] == 18 / 128
    assert sum(d > 0 for d in r["s8"]["differences"]) == 5
    assert r["curve"]["verdict"] == "inconclusive" and 0.006 < r["curve"]["gain"] < 0.007


def test_c49_H3_every_certified_through_9_hybrid_is_over_25_times_a_table_alone_and_breaks_even_near_1_bit_a_weight():
    nets = _readout()["nets"]
    assert len(nets) == 14
    for n in nets:
        d = n["description"]
        hybrid = hybrid_bits(d["params"], BITS_PER_PARAM, d["entries"], d["positions"], ACTIONS)
        table = hybrid_bits(0, BITS_PER_PARAM, d["positions"], d["positions"], ACTIONS)
        assert hybrid["total"] == d["bits"] and hybrid["total"] > 25 * table["total"]
        break_even = (table["total"] - hybrid["table"]) / d["params"]
        assert 0.8 < break_even < 1.1


def test_c49_H3_needs_fewer_exceptions_than_h2_on_6_of_7_seeds():
    r = _as_run()
    h2 = [d["entries"] for d in r["description"]["h2"]]
    h3 = [d["entries"] for d in r["description"]["h3"]]
    assert sum(b < a for a, b in zip(h2, h3)) == 6 and sum(h2) / 7 > 278 and sum(h3) / 7 < 254
