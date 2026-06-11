"""Typed run configuration: the --config-json contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

N_ESTIMATORS_RANGE = (10, 1000)
LEARNING_RATE_RANGE = (0.005, 0.5)
MAX_DEPTH_RANGE = (1, 8)
SUBSAMPLE_RANGE = (0.3, 1.0)


@dataclass(frozen=True)
class TrainerConfig:
    n_estimators: int = 150
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    config: TrainerConfig
    checkpoint: str


def validate_config(config: TrainerConfig) -> None:
    if not N_ESTIMATORS_RANGE[0] <= config.n_estimators <= N_ESTIMATORS_RANGE[1]:
        raise ValueError(f"n_estimators must be in {N_ESTIMATORS_RANGE}, got {config.n_estimators}")
    if not LEARNING_RATE_RANGE[0] <= config.learning_rate <= LEARNING_RATE_RANGE[1]:
        raise ValueError(f"learning_rate must be in {LEARNING_RATE_RANGE}, got {config.learning_rate}")
    if not MAX_DEPTH_RANGE[0] <= config.max_depth <= MAX_DEPTH_RANGE[1]:
        raise ValueError(f"max_depth must be in {MAX_DEPTH_RANGE}, got {config.max_depth}")
    if not SUBSAMPLE_RANGE[0] <= config.subsample <= SUBSAMPLE_RANGE[1]:
        raise ValueError(f"subsample must be in {SUBSAMPLE_RANGE}, got {config.subsample}")
    if not isinstance(config.seed, int):
        raise ValueError(f"seed must be an int, got {config.seed!r}")


def load_config(path: Path) -> TrainerConfig:
    return _config_from_raw(_read_config_object(path))


def load_eval_config(path: Path) -> EvalConfig:
    raw = _read_config_object(path)
    checkpoint = raw.pop("checkpoint", None)
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("evaluate config requires a non-empty 'checkpoint' string")
    raw.pop("eval_episodes", None)
    return EvalConfig(config=_config_from_raw(raw), checkpoint=checkpoint)


def _read_config_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object, got {type(raw).__name__}")
    return raw


def _config_from_raw(raw: dict[str, Any]) -> TrainerConfig:
    known = {f.name for f in fields(TrainerConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key in ("n_estimators", "max_depth", "seed"):
        if key in raw:
            raw[key] = int(raw[key])
    config = TrainerConfig(**raw)
    validate_config(config)
    return config


def config_hash(config: TrainerConfig) -> str:
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
