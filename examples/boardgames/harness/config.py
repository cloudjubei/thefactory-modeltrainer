"""Typed run configuration — the --config-json contract, mirroring examples/cartpole's config.py."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GAME_CHOICES = ("connect4",)
MODEL_NAME_CHOICES = ("random", "heuristic", "mcts")
OPPONENT_CHOICES = ("random", "heuristic", "mcts")
MCTS_SIMS_RANGE = (1, 5000)
EVAL_GAMES_RANGE = (2, 5000)

# §C cost accounting — documented estimate constants (a local run has no invoice, so energy/$ are ESTIMATES).
WATTS_PER_CORE = 12.0  # rough sustained draw of one busy CPU core; override per machine if you measure it.
PRICE_PER_KWH = 0.30  # USD; a generic grid price for the $ estimate.

# The `mcts` opponent rung is a FIXED-strength reference so the ladder (and the locked TEST rung) is decoupled
# from the model's own `mcts_sims` knob — otherwise the test opponent would scale with the model and win-rate
# against it would sit at ~0.5 by construction, making the held-out test meaningless.
OPPONENT_MCTS_SIMS = 120


@dataclass(frozen=True)
class TrainerConfig:
    game: str = "connect4"
    model_name: str = "mcts"
    mcts_sims: int = 80
    opponent: str = "random"
    eval_games: int = 40
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    config: TrainerConfig
    checkpoint: str
    eval_games: int


def validate_config(config: TrainerConfig) -> None:
    if config.game not in GAME_CHOICES:
        raise ValueError(f"game must be one of {GAME_CHOICES}, got {config.game!r}")
    if config.model_name not in MODEL_NAME_CHOICES:
        raise ValueError(f"model_name must be one of {MODEL_NAME_CHOICES}, got {config.model_name!r}")
    if config.opponent not in OPPONENT_CHOICES:
        raise ValueError(f"opponent must be one of {OPPONENT_CHOICES}, got {config.opponent!r}")
    if not MCTS_SIMS_RANGE[0] <= config.mcts_sims <= MCTS_SIMS_RANGE[1]:
        raise ValueError(f"mcts_sims must be in {MCTS_SIMS_RANGE}, got {config.mcts_sims}")
    if not EVAL_GAMES_RANGE[0] <= config.eval_games <= EVAL_GAMES_RANGE[1]:
        raise ValueError(f"eval_games must be in {EVAL_GAMES_RANGE}, got {config.eval_games}")
    if not isinstance(config.seed, int):
        raise ValueError(f"seed must be an int, got {config.seed!r}")


def load_config(path: Path) -> TrainerConfig:
    return _config_from_raw(_read_config_object(path))


def load_eval_config(path: Path) -> EvalConfig:
    raw = _read_config_object(path)
    checkpoint = raw.pop("checkpoint", None)
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("evaluate config requires a non-empty 'checkpoint' string")
    eval_games = int(raw.pop("eval_games", TrainerConfig.eval_games))
    return EvalConfig(config=_config_from_raw(raw), checkpoint=checkpoint, eval_games=eval_games)


def _read_config_object(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object, got {type(raw).__name__}")
    return raw


def _config_from_raw(raw: dict[str, Any]) -> TrainerConfig:
    raw = dict(raw)
    known = {f.name for f in fields(TrainerConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for int_key in ("mcts_sims", "eval_games", "seed"):
        if int_key in raw:
            raw[int_key] = int(raw[int_key])
    config = TrainerConfig(**raw)
    validate_config(config)
    return config


def config_hash(config: TrainerConfig) -> str:
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def provenance_block(config: TrainerConfig) -> dict[str, Any]:
    """§C.9 the reproducibility tuple validateRunProvenance flags on: gitCommit / configHash / seed / dataVersion."""
    return {
        "ranAt": datetime.now(timezone.utc).isoformat(),
        "configHash": config_hash(config),
        "gitCommit": _git_commit(),
        "seed": config.seed,
        "dataVersion": f"{config.game}-rules-v1",
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parent),
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
