"""Held-out evaluation: deterministic episodes, mean/std return, health flags."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from stable_baselines3.common.base_class import BaseAlgorithm

from trainer.model import build_env

DEGENERATE_ACTION_SHARE = 0.97


@dataclass(frozen=True)
class EvalResult:
    return_mean: float
    return_std: float
    episodes: int
    flags: list[str] = field(default_factory=list)


def evaluate_model(model: BaseAlgorithm, seed: int, episodes: int = 20) -> EvalResult:
    env = build_env()
    returns: list[float] = []
    action_counts: Counter[int] = Counter()
    total_steps = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            total_steps += 1
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_return += float(reward)
            done = terminated or truncated
        returns.append(episode_return)
    env.close()

    mean = sum(returns) / len(returns)
    std = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
    flags: list[str] = []
    if any(math.isnan(r) for r in returns) or math.isnan(mean) or math.isnan(std):
        flags.append("nan_metrics")
    if total_steps and max(action_counts.values()) / total_steps > DEGENERATE_ACTION_SHARE:
        flags.append("degenerate_policy")
    return EvalResult(return_mean=mean, return_std=std, episodes=episodes, flags=flags)
