"""Typed run configuration: the --config-json contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ALGO_CHOICES = ("ppo", "dqn")
NET_ARCH_CHOICES = ("64,64", "256,256")
LEARNING_RATE_RANGE = (0.00001, 0.01)
GAMMA_RANGE = (0.8, 0.9999)
TOTAL_TIMESTEPS_RANGE = (1000, 200000)


@dataclass(frozen=True)
class TrainerConfig:
    algo: str = "ppo"
    learning_rate: float = 0.0003
    gamma: float = 0.99
    total_timesteps: int = 8000
    net_arch: str = "64,64"
    seed: int = 0


def validate_config(config: TrainerConfig) -> None:
    if config.algo not in ALGO_CHOICES:
        raise ValueError(f"algo must be one of {ALGO_CHOICES}, got {config.algo!r}")
    if config.net_arch not in NET_ARCH_CHOICES:
        raise ValueError(f"net_arch must be one of {NET_ARCH_CHOICES}, got {config.net_arch!r}")
    if not LEARNING_RATE_RANGE[0] <= config.learning_rate <= LEARNING_RATE_RANGE[1]:
        raise ValueError(f"learning_rate must be in {LEARNING_RATE_RANGE}, got {config.learning_rate}")
    if not GAMMA_RANGE[0] <= config.gamma <= GAMMA_RANGE[1]:
        raise ValueError(f"gamma must be in {GAMMA_RANGE}, got {config.gamma}")
    if not TOTAL_TIMESTEPS_RANGE[0] <= config.total_timesteps <= TOTAL_TIMESTEPS_RANGE[1]:
        raise ValueError(f"total_timesteps must be in {TOTAL_TIMESTEPS_RANGE}, got {config.total_timesteps}")
    if not isinstance(config.seed, int):
        raise ValueError(f"seed must be an int, got {config.seed!r}")


def load_config(path: Path) -> TrainerConfig:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object, got {type(raw).__name__}")
    known = {f.name for f in fields(TrainerConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "total_timesteps" in raw:
        raw["total_timesteps"] = int(raw["total_timesteps"])
    if "seed" in raw:
        raw["seed"] = int(raw["seed"])
    config = TrainerConfig(**raw)
    validate_config(config)
    return config


def config_hash(config: TrainerConfig) -> str:
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
