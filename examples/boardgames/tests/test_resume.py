"""Direct tests for harness/resume.py — the preflight that stands between an interrupted run and a dishonest one."""
from __future__ import annotations

import json

import pytest
import torch

from harness.ledger import run_budget
from harness.resume import (
    buffer_status,
    era_status,
    ledger_readable_prefix,
    orphan_checkpoints,
    preflight,
)
from harness.scaled_run import _completed_batches


def _ckpt(run_dir, batch: int) -> None:
    torch.save({"state_dict": {}, "arch": {"channels": 8}, "channels": 8}, run_dir / f"ckpt_{batch}.pt")


def _metrics(run_dir, batches) -> None:
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps({"batch": b, "games": 400}) + "\n" for b in batches))


def _run(tmp_path, ckpts, metric_batches, game: str = "othello", stamp: str | None = "MATCH"):
    run_dir = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    for b in ckpts:
        _ckpt(run_dir, b)
    if metric_batches is not None:
        _metrics(run_dir, metric_batches)
    if stamp is not None:
        from harness.fingerprint import training_fingerprint

        code = training_fingerprint(game) if stamp == "MATCH" else stamp
        (run_dir / "provenance.json").write_text(json.dumps({"training_fingerprint": code, "game": game}))
    return run_dir


def _good_buffer(run_dir) -> None:
    torch.save([(torch.randn(2, 8, 8), [0.1] * 65, -0.5)], run_dir / "buffer.pt")


def test_era_status_passes_when_the_training_code_has_not_moved(tmp_path):
    st = era_status(_run(tmp_path, [0], [0]))
    assert st["ok"] and st["now"] == st["stamped"]


def test_era_status_refuses_a_run_whose_training_code_moved_underneath_it(tmp_path):
    st = era_status(_run(tmp_path, [0], [0], stamp="deadbeefcafe"))
    assert not st["ok"]
    assert "deadbeefcafe" in st["reason"] and st["now"] in st["reason"]


def test_era_status_refuses_an_unstamped_run_that_already_has_checkpoints(tmp_path):
    st = era_status(_run(tmp_path, [0, 1], [0, 1], stamp=None))
    assert not st["ok"] and "provenance" in st["reason"]


def test_era_status_allows_a_fresh_unstamped_run_dir(tmp_path):
    st = era_status(_run(tmp_path, [], None, stamp=None))
    assert st["ok"]


def test_era_status_follows_the_run_s_own_game(tmp_path):
    from harness.fingerprint import training_fingerprint

    st = era_status(_run(tmp_path, [0], [0], game="connect4"))
    assert st["ok"] and st["now"] == training_fingerprint("connect4")


def test_orphan_checkpoints_finds_a_checkpoint_written_before_its_metrics_row(tmp_path):
    assert orphan_checkpoints(_run(tmp_path, [0, 1], [0])) == [1]


def test_orphan_checkpoints_is_empty_when_every_checkpoint_has_its_row(tmp_path):
    assert orphan_checkpoints(_run(tmp_path, [0, 1, 2], [0, 1, 2])) == []


def test_orphan_checkpoints_condemns_everything_above_a_gap(tmp_path):
    assert orphan_checkpoints(_run(tmp_path, [0, 1, 2], [0, 2])) == [1, 2]


def test_ledger_readable_prefix_is_the_contiguous_run_both_agree_on(tmp_path):
    assert ledger_readable_prefix(_run(tmp_path, [0, 1, 2], [0, 2])) == 1
    assert ledger_readable_prefix(_run(tmp_path, [0, 1, 2], [0, 1, 2])) == 3
    assert ledger_readable_prefix(_run(tmp_path, [], None)) == 0


def test_buffer_status_reports_absent_ok_and_corrupt(tmp_path):
    run_dir = _run(tmp_path, [0], [0])
    assert buffer_status(run_dir) == "absent"
    _good_buffer(run_dir)
    assert buffer_status(run_dir) == "ok"
    blob = (run_dir / "buffer.pt").read_bytes()
    (run_dir / "buffer.pt").write_bytes(blob[: len(blob) // 2])
    assert buffer_status(run_dir) == "corrupt"


def test_preflight_reports_a_clean_run_as_resumable(tmp_path):
    run_dir = _run(tmp_path, [0, 1], [0, 1])
    _good_buffer(run_dir)
    r = preflight(run_dir)
    assert r["ok"] and r["resume_at"] == 2 and r["orphans"] == [] and r["buffer"] == "ok"


def test_preflight_refuses_without_repairing_when_asked_only_to_look(tmp_path):
    run_dir = _run(tmp_path, [0, 1], [0])
    r = preflight(run_dir)
    assert not r["ok"] and r["orphans"] == [1]
    assert (run_dir / "ckpt_1.pt").exists(), "a read-only preflight must not delete anything"


def test_repair_deletes_the_orphan_so_the_batch_is_retrained(tmp_path):
    run_dir = _run(tmp_path, [0, 1], [0])
    r = preflight(run_dir, repair=True)
    assert r["ok"] and r["repaired"] == ["ckpt_1.pt"]
    assert not (run_dir / "ckpt_1.pt").exists()
    assert (run_dir / "ckpt_0.pt").exists(), "repair must keep the checkpoints that ARE sound"
    assert _completed_batches(run_dir) == [0] and r["resume_at"] == 1


def test_every_checkpoint_surviving_repair_is_ledger_readable(tmp_path):
    """The property the guard exists for: after repair, `run_budget` succeeds on each remaining checkpoint."""
    run_dir = _run(tmp_path, [0, 1, 2, 3], [0, 1])
    preflight(run_dir, repair=True)
    survivors = _completed_batches(run_dir)
    assert survivors == [0, 1]
    for b in survivors:
        assert run_budget(run_dir / f"ckpt_{b}.pt")["batch"] == b


def test_repair_truncates_metrics_rows_left_above_the_gap(tmp_path):
    """Rows above the gap must go too: retraining would append a SECOND row for that batch and `run_budget`
    counts rows, so a stale row makes every later checkpoint unreadable all over again."""
    run_dir = _run(tmp_path, [0, 1, 2], [0, 2])
    preflight(run_dir, repair=True)
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines() if x.strip()]
    assert [r["batch"] for r in rows] == [0]


def test_repair_deletes_a_corrupt_buffer_but_keeps_a_sound_one(tmp_path):
    run_dir = _run(tmp_path, [0], [0])
    _good_buffer(run_dir)
    sound = (run_dir / "buffer.pt").read_bytes()
    preflight(run_dir, repair=True)
    assert (run_dir / "buffer.pt").read_bytes() == sound, "a sound buffer is the run's replay history — keep it"
    (run_dir / "buffer.pt").write_bytes(sound[: len(sound) // 2])
    r = preflight(run_dir, repair=True)
    assert not (run_dir / "buffer.pt").exists() and "buffer.pt" in r["repaired"]


def test_repair_never_papers_over_a_moved_training_era(tmp_path):
    """Code drift is a JUDGEMENT, not a repair: deleting checkpoints would not make the eras match."""
    run_dir = _run(tmp_path, [0, 1], [0], stamp="deadbeefcafe")
    r = preflight(run_dir, repair=True)
    assert not r["ok"] and not r["era"]["ok"]
    assert (run_dir / "provenance.json").exists()


def test_preflight_counts_resumes_so_the_lost_optimizer_state_is_on_the_record(tmp_path):
    """`scaled_run` keeps ONE Adam per PROCESS (§C.14), so every resume silently restarts the optimizer's
    moments. That is a real trajectory difference, so the run must at least carry how often it happened."""
    run_dir = _run(tmp_path, [0], [0])
    assert preflight(run_dir, repair=True)["resumes"] == 1
    assert preflight(run_dir, repair=True)["resumes"] == 2
    assert json.loads((run_dir / "provenance.json").read_text())["resumes"] == 2


def test_a_fresh_run_is_not_counted_as_a_resume(tmp_path):
    run_dir = _run(tmp_path, [], None, stamp=None)
    assert preflight(run_dir, repair=True)["resumes"] == 0


@pytest.mark.slow
def test_a_killed_othello_run_resumes_at_the_next_batch_with_its_buffer(tmp_path):
    """End-to-end on the game the transfer run is actually training: batch 0 alone, then a SECOND invocation
    (a fresh process is what an interruption leaves behind) must resume at batch 1, not restart at 0."""
    from harness.scaled_run import run_scaled_experiment

    cfg = {"game": "othello", "run_dir": str(tmp_path / "oth"),
           "net_arch": {"channels": 8, "blocks": 1, "residual": True, "batchnorm": True, "head_hidden": 8},
           "seed": 0, "sims": 4, "iters_per_batch": 1, "games": 2, "epochs": 1, "buffer_cap": 500,
           "opening_plies": 0, "benchmark_positions": 0, "offline_openings": 0, "gate_roots": 0,
           "endgame": False, "league": False, "batches": 1}
    run_scaled_experiment(cfg)
    run_dir = tmp_path / "oth"
    assert _completed_batches(run_dir) == [0] and (run_dir / "buffer.pt").exists()

    seen = []
    run_scaled_experiment({**cfg, "batches": 2}, on_progress=seen.append)
    resume = next(p for p in seen if p["phase"] == "resume")
    assert resume["completed_batches"] == [0] and resume["start_batch"] == 1
    assert _completed_batches(run_dir) == [0, 1]
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines() if x.strip()]
    assert [r["batch"] for r in rows] == [0, 1], "resume must APPEND, never duplicate a batch row"
    assert run_budget(run_dir / "ckpt_1.pt")["batch"] == 1

    # `run_scaled_experiment` stamps no era of its own (the launcher does, from outside the fingerprinted
    # modules), so a run dir carrying checkpoints and no stamp is exactly the case the preflight must refuse.
    assert not preflight(run_dir)["ok"]
    from harness.fingerprint import training_fingerprint

    (run_dir / "provenance.json").write_text(json.dumps(
        {"training_fingerprint": training_fingerprint("othello"), "game": "othello"}))
    stamped = preflight(run_dir)
    assert stamped["ok"] and stamped["resume_at"] == 2 and stamped["buffer"] == "ok"
