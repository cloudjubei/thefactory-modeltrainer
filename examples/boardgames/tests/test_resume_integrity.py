"""§C.22 ROOT FIXES — the three crash-window defects fixed in the training path itself, not just detected.

`tests/test_resume.py` covers the preflight that DETECTS and repairs these from outside. These tests cover the
training path no longer creating them: an interrupted batch must leave no orphan, a killed buffer write must
leave the previous buffer intact, and a resume must not silently restart the optimizer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from harness.ledger import run_budget
from harness.resume import ledger_readable_prefix, preflight
from harness.scaled_run import _save_atomic, run_scaled_experiment

CFG = {"game": "othello",
       "net_arch": {"channels": 8, "blocks": 1, "residual": True, "batchnorm": True, "head_hidden": 8},
       "seed": 0, "sims": 4, "iters_per_batch": 1, "games": 2, "epochs": 1, "buffer_cap": 500,
       "opening_plies": 0, "benchmark_positions": 0, "offline_openings": 0, "gate_roots": 0,
       "endgame": False, "league": False}


def _cfg(tmp_path, batches: int) -> dict:
    return {**CFG, "run_dir": str(tmp_path / "run"), "batches": batches}


def _batches_in(run_dir) -> list[int]:
    return [json.loads(x)["batch"] for x in (run_dir / "metrics.jsonl").read_text().splitlines() if x.strip()]


def test_save_atomic_leaves_the_previous_file_intact_when_the_write_dies(tmp_path, monkeypatch):
    """R3: `torch.save` straight onto buffer.pt left a truncated file that crashed EVERY later resume."""
    path = tmp_path / "buffer.pt"
    _save_atomic([1, 2, 3], path)
    good = path.read_bytes()

    def boom(obj, f):
        f.write(b"half a buffer")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        _save_atomic([4, 5, 6], path)
    assert path.read_bytes() == good, "the committed file must never be touched by a failed write"
    assert list(tmp_path.glob("*.tmp")) == [], "a failed write must not leave its scratch file behind"


def test_save_atomic_roundtrips(tmp_path):
    _save_atomic({"a": torch.zeros(3)}, tmp_path / "x.pt")
    assert torch.load(tmp_path / "x.pt")["a"].shape == (3,)


@pytest.mark.slow
def test_the_run_writes_its_buffer_and_optimizer_through_the_atomic_path(tmp_path, monkeypatch):
    """Wiring, not helper behaviour: `_save_atomic` being correct is worth nothing if the batch loop still
    calls `torch.save` straight onto buffer.pt. Proving the helper works does not prove it is USED."""
    import harness.scaled_run as sr

    seen = []
    real = sr._save_atomic
    monkeypatch.setattr(sr, "_save_atomic", lambda obj, path: (seen.append(Path(path).name), real(obj, path))[1])
    run_scaled_experiment(_cfg(tmp_path, 1))
    assert "buffer.pt" in seen and "opt.pt" in seen


@pytest.mark.slow
def test_an_orphan_checkpoint_is_retrained_rather_than_stranding_the_run(tmp_path):
    """R2: a kill between `save_net` and the metrics append used to lose that row forever — resume skipped past
    the batch and `run_budget` then refused it and everything above it."""
    run_scaled_experiment(_cfg(tmp_path, 1))
    run_dir = tmp_path / "run"
    torch.save({"state_dict": {}, "arch": {"channels": 8}, "channels": 8}, run_dir / "ckpt_1.pt")  # the orphan

    run_scaled_experiment(_cfg(tmp_path, 2))
    assert _batches_in(run_dir) == [0, 1], "batch 1 must be retrained and appear EXACTLY once"
    assert ledger_readable_prefix(run_dir) == 2
    for b in (0, 1):
        assert run_budget(run_dir / f"ckpt_{b}.pt")["batch"] == b


@pytest.mark.slow
def test_a_rerun_batch_replaces_its_metrics_row_instead_of_duplicating_it(tmp_path):
    run_scaled_experiment(_cfg(tmp_path, 2))
    run_dir = tmp_path / "run"
    (run_dir / "ckpt_1.pt").unlink()  # force batch 1 to be retrained while its row is already on disk
    run_scaled_experiment(_cfg(tmp_path, 2))
    assert _batches_in(run_dir) == [0, 1]


@pytest.mark.slow
def test_the_optimizer_survives_a_resume(tmp_path):
    """§C.14 keeps ONE Adam for the run, but nothing persisted it, so each resume silently restarted the moment
    estimates. The property: a run split by an interruption reaches the same optimizer step count as one that
    was never interrupted."""
    straight = tmp_path / "straight"
    run_scaled_experiment({**CFG, "run_dir": str(straight), "batches": 2})
    uninterrupted = _adam_steps(straight)

    split = tmp_path / "run"
    run_scaled_experiment(_cfg(tmp_path, 1))
    run_scaled_experiment(_cfg(tmp_path, 2))  # a SECOND process — what an interruption leaves behind
    assert (split / "opt.pt").exists()
    assert _adam_steps(split) == uninterrupted > 0


def _adam_steps(run_dir) -> int:
    blob = torch.load(run_dir / "opt.pt")
    steps = [s["step"] for s in blob["state"].values() if "step" in s]
    return int(max(float(s) for s in steps)) if steps else 0


@pytest.mark.slow
def test_a_resumed_run_stays_preflight_clean(tmp_path):
    run_scaled_experiment(_cfg(tmp_path, 1))
    run_scaled_experiment(_cfg(tmp_path, 2))
    run_dir = tmp_path / "run"
    from harness.fingerprint import training_fingerprint

    (run_dir / "provenance.json").write_text(json.dumps(
        {"training_fingerprint": training_fingerprint("othello"), "game": "othello", "request": {"batches": 2}}))
    r = preflight(run_dir)
    assert r["ok"] and r["orphans"] == [] and r["buffer"] == "ok"
