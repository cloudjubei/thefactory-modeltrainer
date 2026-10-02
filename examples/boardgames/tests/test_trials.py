"""Direct tests for harness/trials.py — the automatic trial log: every run appends a start line and then an end line
(completed or failed), so a run that was killed outright is the start line with no end."""
from __future__ import annotations

import json

import pytest

from harness.trials import read_trials, trial


def test_a_completed_run_appends_a_start_line_and_an_end_line(tmp_path):
    log = tmp_path / "trials.jsonl"
    with trial("small_floor", ["--arm", "a"], log):
        pass
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["event"] for x in lines] == ["start", "end"] and lines[1]["status"] == "completed"
    assert lines[0]["id"] == lines[1]["id"] and lines[0]["name"] == "small_floor" and lines[0]["argv"] == ["--arm", "a"]


def test_a_run_that_raises_is_logged_as_failed_with_its_error_and_the_error_still_propagates(tmp_path):
    log = tmp_path / "trials.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        with trial("x", [], log):
            raise RuntimeError("boom")
    end = json.loads(log.read_text().splitlines()[-1])
    assert end["status"] == "failed" and "boom" in end["error"]


def test_a_run_stopped_by_systemexit_is_logged_as_failed(tmp_path):
    log = tmp_path / "trials.jsonl"
    with pytest.raises(SystemExit):
        with trial("x", [], log):
            raise SystemExit("evidence not written")
    assert json.loads(log.read_text().splitlines()[-1])["status"] == "failed"


def test_runs_append_and_a_start_without_an_end_reads_as_aborted(tmp_path):
    log = tmp_path / "trials.jsonl"
    with trial("first", [], log):
        pass
    log.write_text(log.read_text() + json.dumps({"event": "start", "id": "z", "name": "killed", "argv": [],
                                                 "at": "2026-10-02T00:00:00+00:00"}) + "\n")
    with trial("third", [], log):
        pass
    runs = read_trials(log)
    assert [(r["name"], r["status"]) for r in runs] == [("first", "completed"), ("killed", "aborted"),
                                                       ("third", "completed")]


def test_a_driver_runs_logged_under_its_own_file_name_with_its_command_line(tmp_path, monkeypatch):
    import sys

    from harness.trials import logged

    log = tmp_path / "trials.jsonl"
    monkeypatch.setattr(sys, "argv", ["scripts/small_floor.py", "--arm", "b"])
    assert logged(lambda: 7, "/x/scripts/small_floor.py", log) == 7
    start = json.loads(log.read_text().splitlines()[0])
    assert start["name"] == "small_floor" and start["argv"] == ["--arm", "b"]
