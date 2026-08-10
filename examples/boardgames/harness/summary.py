"""Builds and writes the RunSummary — the machine-readable result the modeltrainer engine ingests.

Carries the §C metric battery (win-rate truth, episode-reward proxy, draw-rate + first-player-seat confound),
a first-class COST block (wall/cpu time, compute, and documented energy/$ estimates), and a sampled REPLAY so
a game can be followed move by move in the viewer.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from harness.config import PRICE_PER_KWH, WATTS_PER_CORE, TrainerConfig, config_hash, provenance_block
from harness.selfplay import EvalResult

MAX_SERIES_POINTS = 200


def _downsample(values: list[float], max_points: int = MAX_SERIES_POINTS) -> list[float]:
    if len(values) <= max_points:
        return [round(v, 4) for v in values]
    stride = (len(values) - 1) / (max_points - 1)
    return [round(values[round(i * stride)], 4) for i in range(max_points)]


def _cost_block(wall_seconds: float, cpu_seconds: float, ev: EvalResult, steps: dict[str, float]) -> dict[str, Any]:
    energy_wh = cpu_seconds / 3600.0 * WATTS_PER_CORE
    cost_usd = energy_wh / 1000.0 * PRICE_PER_KWH
    return {
        "wallSeconds": round(wall_seconds, 3),
        "cpuSeconds": round(cpu_seconds, 3),
        "gamesPlayed": ev.games,
        "movesEvaluated": ev.total_moves,
        "mctsSimsTotal": ev.sims_used,
        "movesPerSecond": round(ev.total_moves / wall_seconds, 1) if wall_seconds > 0 else 0.0,
        "estEnergyWh": round(energy_wh, 4),
        "estCostUsd": round(cost_usd, 6),
        "assumptions": {"wattsPerCore": WATTS_PER_CORE, "pricePerKwh": PRICE_PER_KWH, "estimated": True},
        "steps": {k: round(v, 3) for k, v in steps.items()},
    }


def build_summary(
    config: TrainerConfig,
    ev: EvalResult,
    checkpoint: str,
    wall_seconds: float,
    cpu_seconds: float,
    steps: dict[str, float],
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "objective": ev.win_rate,
        "metrics": {
            "win_rate": ev.win_rate,
            f"win_rate_vs_{ev.opponent}": ev.win_rate,
            "episode_reward": ev.episode_reward,
            "draw_rate": ev.draw_rate,
            "loss_rate": ev.loss_rate,
            "first_player_seat_winrate": ev.first_player_seat_winrate,
            "games": ev.games,
            "moves_evaluated": ev.total_moves,
            "mcts_sims_total": ev.sims_used,
        },
        "series": {
            "win_rate": _downsample(ev.win_rate_series),
            "episode_reward": _downsample(ev.reward_series),
        },
        "cost": _cost_block(wall_seconds, cpu_seconds, ev, steps),
        "health": {"status": "degenerate" if ev.flags else "ok", "flags": ev.flags},
        "seed": config.seed,
        "config": asdict(config),
        "provenance": provenance_block(config),
        "artifacts": {"checkpoint": checkpoint, "best": False},
    }
    if ev.sample_game is not None:
        summary["sample_game"] = ev.sample_game
    if calibration is not None:
        summary["calibration"] = calibration
    return summary


def build_eval_summary(
    config: TrainerConfig,
    ev: EvalResult,
    checkpoint: str,
    wall_seconds: float,
    cpu_seconds: float,
    steps: dict[str, float],
) -> dict[str, Any]:
    summary = build_summary(config, ev, checkpoint, wall_seconds, cpu_seconds, steps)
    summary["evaluation"] = {"checkpoint": checkpoint, "games": ev.games, "opponent": ev.opponent}
    return summary


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
