"""Direct tests for harness/kalah_paper.py — the published Kalah results (Irving, Donkers & Uiterwijk 2000) and the
judge that reads a check of games/kalah.py against them: the solved values agree with Tables 9 and 10, and the paper's
perfect games replay under its stated rules better than under any other rule reading."""
from __future__ import annotations

import pytest

from harness.kalah_paper import PERFECT_GAMES, SPEC, TABLE_10, paper_report

TEST_SPEC = {"measurement_fp": "m" * 12, "solve": [[1, 1], [1, 2], [4, 1]],
             "variants": ["paper", "skip_start", "nonempty_capture"], "inconsistent": {"4,6": [9, 1]}}


def _replays(broken=()):
    rows = []
    for (m, n), (margin, line) in sorted(PERFECT_GAMES.items()):
        if (m, n) == (4, 6) or (m, n) in broken:
            rows.append({"holes": m, "counters": n, "result": "extra-move mismatch", "margin": None, "turn": 9,
                         "move": 1})
        else:
            rows.append({"holes": m, "counters": n, "result": "end", "margin": margin, "turn": None, "move": None})
    return rows


def _evidence(solved=None, paper=None, others=((4, 1), (5, 1))):
    return {"measurement_fingerprint": "m" * 12,
            "solved": solved if solved is not None else [
                {"holes": 1, "counters": 1, "margin": 0}, {"holes": 1, "counters": 2, "margin": -2},
                {"holes": 4, "counters": 1, "margin": 2}],
            "replays": {"paper": paper if paper is not None else _replays(),
                        "skip_start": _replays(broken=[others[0]]), "nonempty_capture": _replays(broken=[others[1]])},
            "repairs": []}


def test_a_check_that_agrees_with_both_tables_supports_both_claims():
    r = paper_report(_evidence(), TEST_SPEC)
    assert r["integrity"] == [] and r["values"] == {"verdict": "supported", "disagreements": []}
    assert r["replays"]["verdict"] == "supported"
    assert r["replays"]["consistent"] == {"paper": 16, "skip_start": 15, "nonempty_capture": 15}


def test_a_wrong_win_draw_loss_or_a_wrong_margin_refutes_the_values():
    e = _evidence(solved=[{"holes": 1, "counters": 1, "margin": 1}, {"holes": 1, "counters": 2, "margin": -2},
                          {"holes": 4, "counters": 1, "margin": 2}])
    r = paper_report(e, TEST_SPEC)["values"]
    assert r["verdict"] == "refuted" and r["disagreements"] == [{"shape": [1, 1], "ours": 1, "paper": "D"}]
    e = _evidence(solved=[{"holes": 1, "counters": 1, "margin": 0}, {"holes": 1, "counters": 2, "margin": -2},
                          {"holes": 4, "counters": 1, "margin": 4}])
    r = paper_report(e, TEST_SPEC)["values"]
    assert r["verdict"] == "refuted" and r["disagreements"] == [{"shape": [4, 1], "ours": 4, "paper": 2}]


def test_a_second_broken_line_under_the_paper_s_rules_refutes_the_replays():
    r = paper_report(_evidence(paper=_replays(broken=[(5, 1)])), TEST_SPEC)["replays"]
    assert r["verdict"] == "refuted" and r["consistent"]["paper"] == 15


def test_the_4_6_line_must_break_where_registered():
    paper = _replays()
    paper[5] = {**paper[5], "turn": 8}
    assert paper_report(_evidence(paper=paper), TEST_SPEC)["replays"]["verdict"] == "refuted"
    paper = _replays()
    paper[5] = {**paper[5], "result": "end", "margin": 0}
    assert paper_report(_evidence(paper=paper), TEST_SPEC)["replays"]["verdict"] == "refuted"


def test_a_line_that_ends_on_the_wrong_margin_is_not_consistent():
    paper = _replays()
    paper[0] = {**paper[0], "margin": paper[0]["margin"] + 2}
    r = paper_report(_evidence(paper=paper), TEST_SPEC)["replays"]
    assert r["consistent"]["paper"] == 15 and r["verdict"] == "refuted"
    paper = _replays()
    paper[0] = {**paper[0], "result": "never ended"}
    assert paper_report(_evidence(paper=paper), TEST_SPEC)["replays"]["consistent"]["paper"] == 15


def test_an_alternative_reading_that_fits_as_well_refutes_the_replays():
    e = _evidence()
    e["replays"]["skip_start"] = _replays()
    r = paper_report(e, TEST_SPEC)["replays"]
    assert r["verdict"] == "refuted" and r["consistent"]["skip_start"] == 16


def test_a_one_character_repair_of_the_4_6_line_refutes_the_replays():
    e = _evidence()
    e["repairs"] = ["1-0-2-3"]
    assert paper_report(e, TEST_SPEC)["replays"]["verdict"] == "refuted"


@pytest.mark.parametrize("breakage", ["fingerprint", "missing shape", "extra shape", "missing variant",
                                      "short replay"])
def test_a_check_that_is_not_the_registered_one_is_not_run(breakage):
    e = _evidence()
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "missing shape":
        e["solved"] = e["solved"][1:]
    elif breakage == "extra shape":
        e["solved"].append({"holes": 2, "counters": 1, "margin": 1})
    elif breakage == "missing variant":
        del e["replays"]["nonempty_capture"]
    else:
        e["replays"]["paper"] = e["replays"]["paper"][1:]
    r = paper_report(e, TEST_SPEC)
    assert r["integrity"] and r["values"] == {"verdict": "not_run"} and r["replays"] == {"verdict": "not_run"}


def test_the_tables_are_the_paper_s():
    assert len(PERFECT_GAMES) == 17 and PERFECT_GAMES[(6, 5)][0] == 12 and PERFECT_GAMES[(4, 6)][0] == 0
    assert TABLE_10[3] == "DWWWWL" and TABLE_10[6] == "WWWWW" and sorted(TABLE_10) == [1, 2, 3, 4, 5, 6]


def test_the_registered_check_solves_every_shape_whose_memo_was_measured_and_reads_four_rule_variants():
    shapes = [[m, n] for m in (1, 2, 3) for n in range(1, 7)] + [[4, 1], [4, 2], [4, 3], [5, 1], [5, 2], [6, 1]]
    assert SPEC["solve"] == shapes
    assert SPEC["variants"] == ["paper", "skip_start", "nonempty_capture", "both"]
    assert SPEC["inconsistent"] == {"4,6": [9, 1]}
