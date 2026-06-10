"""Training loop with wall-clock throughput measurement."""
from __future__ import annotations

import time
from dataclasses import dataclass

from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass(frozen=True)
class TrainResult:
    seconds: float
    fps: float


def train_model(model: BaseAlgorithm, total_timesteps: int) -> TrainResult:
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    seconds = time.perf_counter() - start
    return TrainResult(seconds=seconds, fps=total_timesteps / seconds)
