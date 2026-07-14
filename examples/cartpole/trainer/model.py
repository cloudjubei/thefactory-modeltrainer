"""Model/policy factory: SB3 PPO or DQN on CartPole-v1."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

from trainer.config import TrainerConfig

ENV_ID = "CartPole-v1"


def parse_net_arch(spec: str) -> list[int]:
    return [int(width) for width in spec.split(",")]


def build_env() -> gym.Env:
    return gym.make(ENV_ID)


def build_train_env() -> Monitor:
    return Monitor(build_env())


def build_model(config: TrainerConfig, env: gym.Env) -> BaseAlgorithm:
    return _algo_class(config.model_name)(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        seed=config.seed,
        policy_kwargs={"net_arch": parse_net_arch(config.net_arch)},
        device="cpu",
        verbose=0,
    )


def load_model(config: TrainerConfig, checkpoint: Path) -> BaseAlgorithm:
    return _algo_class(config.model_name).load(str(checkpoint), device="cpu")


def _algo_class(model_name: str) -> type[BaseAlgorithm]:
    return PPO if model_name == "ppo" else DQN
