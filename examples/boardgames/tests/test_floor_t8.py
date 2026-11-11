"""Direct tests for harness/floor_t8.py — T8, T5's game written again with references and exception flags read from
the position, and the judgement of a file that must read back, win every line and be smaller."""
from __future__ import annotations

from harness.floor_t5 import SPEC as T5
from harness.floor_t8 import SPEC, t8_report

JUDGED = {**SPEC, "control_bits": 1_000}
SOURCE = {"sha256": "t5", "bits": 1_000}


def test_the_spec_writes_t5_s_game_under_its_rule():
    assert SPEC["rule"] == T5["rule"] and SPEC["source"] == T5["file"] == "c49_t5.json.gz"
    assert SPEC["control_bits"] == 109_034 and SPEC["file"] == "c49_t8.json.gz"
    assert SPEC["support"] == (99, 100) and SPEC["refute"] == (999, 1000)


def _evidence(bits=900, won=True, leaves=7, sha="t5"):
    return {"spec": "harness.floor_t8", "bits": bits, "source_sha256": sha,
            "replay": {"won": won, "leaves": leaves, "reason": None}}


def _report(e=None, read=None, source=SOURCE, spec=JUDGED):
    e = e or _evidence()
    return t8_report(e, read or {"bits": e["bits"], "differences": 0, "leaves": 7}, source, spec)


def test_a_file_that_reads_back_and_wins_every_line_is_decoded():
    assert _report() == {"decoded": "supported", "size": "supported"}


def test_a_file_that_reads_back_differently_refutes():
    assert _report(read={"bits": 900, "differences": 1, "leaves": 7}) == {"decoded": "refuted", "size": "not_run"}


def test_a_file_of_another_length_refutes():
    assert _report(read={"bits": 901, "differences": 0, "leaves": 7})["decoded"] == "refuted"
    assert _report(read={"bits": None, "differences": 0, "leaves": 7})["decoded"] == "refuted"


def test_a_lost_replay_refutes_and_has_no_size():
    assert _report(_evidence(won=False, bits=10)) == {"decoded": "refuted", "size": "not_run"}


def test_a_replay_of_other_leaves_refutes():
    assert _report(_evidence(leaves=6))["decoded"] == "refuted"


def test_the_size_bars_are_inclusive():
    assert _report(_evidence(bits=990))["size"] == "supported"
    assert _report(_evidence(bits=991))["size"] == "inconclusive"
    assert _report(_evidence(bits=998))["size"] == "inconclusive"
    assert _report(_evidence(bits=999))["size"] == "refuted"


def test_a_run_off_its_spec_or_its_source_is_not_run():
    neither = {"decoded": "not_run", "size": "not_run"}
    assert _report({**_evidence(), "spec": "harness.floor_t5"}) == neither
    assert _report(_evidence(sha="other")) == neither
    assert _report(source={**SOURCE, "bits": 999}) == neither
