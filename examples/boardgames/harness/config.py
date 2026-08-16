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
MODEL_NAME_CHOICES = ("random", "heuristic", "mcts", "alphazero")
# `oracle_depth` = the depth-limited near-perfect solver (harness/solver.py) — a fast, tactically-perfect
# reference opponent (the exact `oracle` is a benchmark labeller, too slow to PLAY from the opening in Python).
OPPONENT_CHOICES = ("random", "heuristic", "mcts", "oracle_depth")
MCTS_SIMS_RANGE = (1, 5000)
MCTS_SOLVE_ENDGAME_RANGE = (0, 30)  # empty-cell threshold for mcts's opt-in exact-endgame cutoff (0 = pure mcts)
ORACLE_DEPTH_RANGE = (1, 20)
BENCHMARK_POSITIONS_RANGE = (0, 500)
BENCHMARK_MIN_MOVES = 20  # score positions with ≥ this many stones down so the exact oracle solve stays fast
BENCHMARK_SEED = 20240812  # FIXED corpus → every model's `oracle_optimality_rate` is scored on the SAME positions
EVAL_GAMES_RANGE = (2, 5000)
AZ_ITERATIONS_RANGE = (1, 100)
AZ_SELFPLAY_GAMES_RANGE = (1, 500)
AZ_SIMS_RANGE = (1, 2000)
AZ_EPOCHS_RANGE = (1, 50)
AZ_DISTILL_RANGE = (0, 2000)
AZ_SOLVE_ENDGAME_RANGE = (0, 30)  # empty-cell threshold for the trained net's opt-in exact-endgame cutoff

# §C cost accounting — documented estimate constants (a local run has no invoice, so energy/$ are ESTIMATES).
WATTS_PER_CORE = 12.0  # rough sustained draw of one busy CPU core; override per machine if you measure it.
PRICE_PER_KWH = 0.30  # USD; a generic grid price for the $ estimate.

# The `mcts` opponent rung is a FIXED-strength reference so the ladder (and the locked TEST rung) is decoupled
# from the model's own `mcts_sims` knob — otherwise the test opponent would scale with the model and win-rate
# against it would sit at ~0.5 by construction, making the held-out test meaningless.
OPPONENT_MCTS_SIMS = 120
# A FIXED-strength reference for the comparable-strength yardstick (`win_rate_vs_strong_mcts`) — frozen so
# that metric is a pairing against a stable rating anchor (mcts@400), comparable across runs + campaigns.
STRONG_MCTS_SIMS = 400


@dataclass(frozen=True)
class TrainerConfig:
    game: str = "connect4"
    model_name: str = "mcts"
    mcts_sims: int = 80
    mcts_solve_endgame: int = 0  # >0 = mcts plays a perfect endgame once ≤ this many cells are empty (0 = pure)
    opponent: str = "random"
    eval_games: int = 40
    seed: int = 0
    oracle_depth: int = 10  # search depth of the `oracle_depth` reference opponent (fixed → a stable rung)
    benchmark_positions: int = 24  # positions scored for `oracle_optimality_rate` (0 = skip the benchmark)
    # alphazero levers (used only when model_name == "alphazero"; pinned n/a otherwise by the engine):
    az_iterations: int = 8
    az_selfplay_games: int = 24
    az_sims: int = 100
    az_epochs: int = 5
    az_warm_start: int = 1  # 1 = warm-start from the champion (cumulative); 0 = train from scratch (reproducible)
    az_distill_positions: int = 96  # oracle-labelled optimal-play examples to imprint each run (0 = no distillation)
    az_solve_endgame: int = 0  # >0 = the trained net plays a PERFECT endgame once ≤ this many cells are empty


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
    if not MCTS_SOLVE_ENDGAME_RANGE[0] <= config.mcts_solve_endgame <= MCTS_SOLVE_ENDGAME_RANGE[1]:
        raise ValueError(f"mcts_solve_endgame must be in {MCTS_SOLVE_ENDGAME_RANGE}, got {config.mcts_solve_endgame}")
    if not ORACLE_DEPTH_RANGE[0] <= config.oracle_depth <= ORACLE_DEPTH_RANGE[1]:
        raise ValueError(f"oracle_depth must be in {ORACLE_DEPTH_RANGE}, got {config.oracle_depth}")
    if not BENCHMARK_POSITIONS_RANGE[0] <= config.benchmark_positions <= BENCHMARK_POSITIONS_RANGE[1]:
        raise ValueError(f"benchmark_positions must be in {BENCHMARK_POSITIONS_RANGE}, got {config.benchmark_positions}")
    if not EVAL_GAMES_RANGE[0] <= config.eval_games <= EVAL_GAMES_RANGE[1]:
        raise ValueError(f"eval_games must be in {EVAL_GAMES_RANGE}, got {config.eval_games}")
    if not isinstance(config.seed, int):
        raise ValueError(f"seed must be an int, got {config.seed!r}")
    if not AZ_ITERATIONS_RANGE[0] <= config.az_iterations <= AZ_ITERATIONS_RANGE[1]:
        raise ValueError(f"az_iterations must be in {AZ_ITERATIONS_RANGE}, got {config.az_iterations}")
    if not AZ_SELFPLAY_GAMES_RANGE[0] <= config.az_selfplay_games <= AZ_SELFPLAY_GAMES_RANGE[1]:
        raise ValueError(f"az_selfplay_games must be in {AZ_SELFPLAY_GAMES_RANGE}, got {config.az_selfplay_games}")
    if not AZ_SIMS_RANGE[0] <= config.az_sims <= AZ_SIMS_RANGE[1]:
        raise ValueError(f"az_sims must be in {AZ_SIMS_RANGE}, got {config.az_sims}")
    if not AZ_EPOCHS_RANGE[0] <= config.az_epochs <= AZ_EPOCHS_RANGE[1]:
        raise ValueError(f"az_epochs must be in {AZ_EPOCHS_RANGE}, got {config.az_epochs}")
    if config.az_warm_start not in (0, 1):
        raise ValueError(f"az_warm_start must be 0 or 1, got {config.az_warm_start}")
    if not AZ_DISTILL_RANGE[0] <= config.az_distill_positions <= AZ_DISTILL_RANGE[1]:
        raise ValueError(f"az_distill_positions must be in {AZ_DISTILL_RANGE}, got {config.az_distill_positions}")
    if not AZ_SOLVE_ENDGAME_RANGE[0] <= config.az_solve_endgame <= AZ_SOLVE_ENDGAME_RANGE[1]:
        raise ValueError(f"az_solve_endgame must be in {AZ_SOLVE_ENDGAME_RANGE}, got {config.az_solve_endgame}")


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
    # Read the declared levers and IGNORE any other key — the engine legitimately injects config it owns
    # (e.g. `device`, checkpoint/continue refs) that a consumer must not hard-fail on.
    known = {f.name for f in fields(TrainerConfig)}
    filtered = {k: v for k, v in raw.items() if k in known}
    for int_key in ("mcts_sims", "mcts_solve_endgame", "oracle_depth", "benchmark_positions", "eval_games", "seed", "az_iterations", "az_selfplay_games", "az_sims", "az_epochs", "az_distill_positions", "az_solve_endgame"):
        if int_key in filtered:
            filtered[int_key] = int(filtered[int_key])
    if "az_warm_start" in filtered:
        v = filtered["az_warm_start"]
        filtered["az_warm_start"] = 1 if (v is True or str(v).strip().lower() in ("1", "true", "yes")) else 0
    config = TrainerConfig(**filtered)
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
