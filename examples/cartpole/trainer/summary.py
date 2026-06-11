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

MAX_SERIES_POINTS = 200


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
        "series": {
            "episode_return": downsample_series(train_result.episode_returns, MAX_SERIES_POINTS),
        },
        "health": _health_block(eval_result),
        "seed": config.seed,
        "config": asdict(config),
        "provenance": _provenance_block(config),
        "artifacts": {"checkpoint": checkpoint, "best": False},
    }
    if calibration is not None:
        summary["calibration"] = calibration
    return summary


def build_eval_summary(config: TrainerConfig, eval_result: EvalResult, checkpoint: str) -> dict[str, Any]:
    return {
        "objective": eval_result.return_mean,
        "metrics": {
            "eval_return_mean": eval_result.return_mean,
            "eval_return_std": eval_result.return_std,
            "episodes_evaluated": eval_result.episodes,
        },
        "health": _health_block(eval_result),
        "seed": config.seed,
        "config": asdict(config),
        "provenance": _provenance_block(config),
        "evaluation": {"checkpoint": checkpoint, "episodes": eval_result.episodes},
    }


def downsample_series(values: list[float], max_points: int) -> list[float]:
    if len(values) <= max_points:
        return list(values)
    stride = (len(values) - 1) / (max_points - 1)
    return [values[round(i * stride)] for i in range(max_points)]


def _health_block(eval_result: EvalResult) -> dict[str, Any]:
    return {"status": "degenerate" if eval_result.flags else "ok", "flags": eval_result.flags}


def _provenance_block(config: TrainerConfig) -> dict[str, Any]:
    return {"ranAt": datetime.now(timezone.utc).isoformat(), "configHash": config_hash(config)}


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
