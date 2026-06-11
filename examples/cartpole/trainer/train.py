"""Training loop with wall-clock throughput measurement."""
from __future__ import annotations

import time
from dataclasses import dataclass

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor


@dataclass(frozen=True)
class TrainResult:
    seconds: float
    fps: float
    episode_returns: list[float]


def train_model(model: BaseAlgorithm, total_timesteps: int, train_env: Monitor) -> TrainResult:
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    seconds = time.perf_counter() - start
    returns = [float(r) for r in train_env.get_episode_rewards()]
    return TrainResult(seconds=seconds, fps=total_timesteps / seconds, episode_returns=returns)
