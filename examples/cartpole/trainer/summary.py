"""Builds and writes the RunSummary: the machine-readable result the orchestrator ingests."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.config import TrainerConfig, config_hash
from trainer.evaluate import EvalResult
from trainer.train import TrainResult


def build_summary(
    config: TrainerConfig,
    train_result: TrainResult,
    eval_result: EvalResult,
    checkpoint: str,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "objective": eval_result.return_mean,
        "metrics": {
            "eval_return_mean": eval_result.return_mean,
            "eval_return_std": eval_result.return_std,
            "train_fps": train_result.fps,
            "train_seconds": train_result.seconds,
            "episodes_evaluated": eval_result.episodes,
        },
        "health": {
            "status": "degenerate" if eval_result.flags else "ok",
            "flags": eval_result.flags,
        },
        "seed": config.seed,
        "config": asdict(config),
        "provenance": {
            "ranAt": datetime.now(timezone.utc).isoformat(),
            "configHash": config_hash(config),
        },
        "artifacts": {"checkpoint": checkpoint, "best": False},
    }
    if calibration is not None:
        summary["calibration"] = calibration
    return summary


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
