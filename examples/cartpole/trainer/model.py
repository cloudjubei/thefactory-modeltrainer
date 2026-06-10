"""Model/policy factory: SB3 PPO or DQN on CartPole-v1."""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from trainer.config import TrainerConfig

ENV_ID = "CartPole-v1"


def parse_net_arch(spec: str) -> list[int]:
    return [int(width) for width in spec.split(",")]


def build_env() -> gym.Env:
    return gym.make(ENV_ID)


def build_model(config: TrainerConfig) -> BaseAlgorithm:
    algo_cls = PPO if config.algo == "ppo" else DQN
    return algo_cls(
        "MlpPolicy",
        build_env(),
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        seed=config.seed,
        policy_kwargs={"net_arch": parse_net_arch(config.net_arch)},
        device="cpu",
        verbose=0,
    )
