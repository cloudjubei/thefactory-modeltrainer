"""§C.7 batched/resumable scaled-run driver — a checkpoint + metrics per batch, and RESUME picks up where it left off."""
import json
from pathlib import Path

from harness.scaled_run import run_scaled_experiment

ARCH = {"channels": 16, "blocks": 1, "residual": True, "batchnorm": True, "head_hidden": 8}


def _req(run_dir, batches):
    return {"game": "connect4", "run_dir": str(run_dir), "net_arch": ARCH, "seed": 0, "sims": 6,
            "iters_per_batch": 1, "games": 3, "epochs": 1, "buffer_cap": 2000, "opening_plies": 2,
            "batches": batches, "benchmark_positions": 6, "offline_openings": 3}


def test_batches_checkpoint_and_resume(tmp_path):
    run_dir = Path(tmp_path) / "run"
    # Batch 0 only.
    r1 = run_scaled_experiment(_req(run_dir, batches=1))
    assert (run_dir / "ckpt_0.pt").exists()
    assert len(r1["metrics"]) == 1 and r1["metrics"][0]["batch"] == 0
    m0 = r1["metrics"][0]
    assert "opening_value" in m0 and "oracle_optimality_rate" in m0 and "offline" in m0
    assert "p1_loss" in m0["offline"]
    lines1 = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines1) == 1

    # RESUME to batch 2: must SKIP batch 0 (checkpoint present) and only run batches 1..1.
    r2 = run_scaled_experiment(_req(run_dir, batches=2))
    assert (run_dir / "ckpt_1.pt").exists()
    batches_seen = [m["batch"] for m in r2["metrics"]]
    assert batches_seen == [0, 1]  # batch 0 read back from metrics.jsonl, batch 1 freshly trained
    lines2 = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines2) == 2  # exactly one NEW line appended on resume (batch 0 not re-run)
    # iterations_done is cumulative across batches
    assert json.loads(lines2[-1])["iterations_done"] == 2
