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
OPPONENT_CHOICES = ("random", "heuristic", "mcts", "oracle_depth", "book")
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
AZ_DISTILL_GAMES_RANGE = (0, 2000)
AZ_SOLVE_ENDGAME_RANGE = (0, 30)  # empty-cell threshold for the trained net's opt-in exact-endgame cutoff
# §C.7 online endgame-tablebase-from-play (#2) — records/solves endgames DURING self-play into a run-owned
# tablebase, memoises them, and injects their EXACT values as training targets so the value head converges to a
# fixed quality target in fewer iterations. All OFF by default (a run stays byte-identical to pure #1).
AZ_ENDGAME_MAX_EMPTY_RANGE = (0, 20)  # empties threshold for the record + cheap-solve cap (cost cliff toward 20)
AZ_ENDGAME_EXTEND_POSITIONS_RANGE = (0, 200000)  # per-iteration retrograde-extension budget (positions proven)
AZ_ENDGAME_EXTEND_SECONDS_RANGE = (0.0, 120.0)  # per-iteration extension deadline (checked between positions)
AZ_ENDGAME_CAP_RANGE = (10000, 50000000)  # run-tablebase capacity = RAM bound (floor guards a useless store)
# §C.7 NET CAPACITY levers — the reproduction floor for strong C4 is 5-19 residual blocks x 128 filters (~1.6M
# params); the legacy default (blocks=0/residual=0) is the ~20K-param net that under-fit. The SAME seam scales the
# net per game (connect4 -> chess/Go) instead of hard-coding a 2-conv shape.
AZ_CHANNELS_RANGE = (8, 512)      # conv filter width
AZ_BLOCKS_RANGE = (0, 40)         # residual blocks (depth); needs az_residual+az_batchnorm to train deep
AZ_HEAD_HIDDEN_RANGE = (0, 1024)  # policy/value head-tower hidden width (0 = bare legacy linear readout)
# A DEPLOYED / eval net plays the endgame with the EXACT solver once ≤ this many cells are empty (provably
# perfect + ~ms). ON by default so a crowned champion never approximates a solvable endgame with the value head;
# self-play keeps it OFF via the AlphaZeroAgent class default so exploration isn't collapsed onto solver moves.
DEFAULT_AZ_SOLVE_ENDGAME = 22

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
    az_distill_positions: int = 96  # late-game oracle-labelled examples to imprint each run (0 = no late distillation)
    az_distill_games: int = 0  # broad OPENING→endgame oracle-distillation games (0 = off); teaches centre-first play
    az_solve_endgame: int = DEFAULT_AZ_SOLVE_ENDGAME  # trained net plays a PERFECT endgame once ≤ this many empty
    # The VALIDATED GENERIC recipe (plan §C.6) — off by default (preserves the classic loop); enable via the manifest.
    az_gumbel: int = 0  # 1 = Gumbel/Sequential-Halving completed-Q search + policy target (the measured deploy/target win)
    az_c_scale: float = 0.1  # completed-Q σ scale (calibrated default; only used when az_gumbel)
    az_value_n_step: int = 0  # n-step/TD value target off a lagged net (0 = raw-MC outcome). NOT a free win: at
    # small net size it bootstraps off a lagged copy whose OWN opening belief is ~0 and MEASURABLY SUPPRESSES opening
    # value (pure-MC +0.38 vs n8 +0.26); keep 0 below ~1M params, re-measure once the value tower can hold a belief.
    az_target_refresh: int = 2  # refresh the lagged value target net every k iterations (only used when az_value_n_step>0)
    az_selfplay_opening_plies: int = 0  # random opening plies per self-play game (0 = canonical) — off-line coverage = ROBUSTNESS
    az_buffer_cap: int = 8000  # replay-buffer size (positions). Raise for LARGE self-play runs so the net trains on a wide history, not just the last ~2 iterations (the AlphaZero sliding window).
    az_pool_frac: float = 0.35  # fraction of self-play games played vs the LEAGUE (past champions + a near-perfect oracle). 0 = PURE self-play (net vs net only, NO solver-derived opponent) — the honest generic test.
    # §C.7 online endgame-tablebase-from-play (#2) — all OFF by default; the master switch alone keeps a run
    # byte-identical to pure #1, AND the loop is only CONSTRUCTED for a game exposing canonical_key + exact_optimal_actions.
    az_endgame_tablebase: int = 0  # master switch. 0 = no run tablebase, learner book=None/solve_endgame=0 (= pure #1)
    az_endgame_max_empty: int = 14  # empties threshold: the learner's record solve_endgame + the extension's cheap-solve cap
    az_endgame_exact_targets: int = 1  # 1 = override self-play value targets with run_tb.proven_value near terminals (inert while empty)
    az_endgame_extend_positions: int = 2000  # per-iteration retrograde-extension budget (positions proven)
    az_endgame_extend_seconds: float = 5.0  # per-iteration extension deadline (thread-safe, checked between positions)
    az_endgame_cap: int = 200000  # run-tablebase capacity = RAM bound (conservative given the OOM history)
    az_endgame_warm_start: int = 0  # 1 = seed run_tb from the committed load_book (default OFF for clean A/B provenance)
    az_endgame_persist: int = 0  # 1 = save the grown run_tb to a RUN-OWNED path at run end (never the committed book)
    # §C.7 net capacity — default = legacy 20K-param net; residual+batchnorm+blocks+head_hidden = the scaled ResNet.
    az_channels: int = 32  # conv filter width (was hard-coded 32)
    az_blocks: int = 0  # residual blocks (depth); >0 requires az_residual
    az_residual: int = 0  # 1 = ResNet tower + head towers; 0 = legacy 2-conv/bare-linear net
    az_batchnorm: int = 0  # 1 = BatchNorm (required to train a deep tower)
    az_head_hidden: int = 0  # policy/value head-tower hidden width (0 = bare linear readout)
    az_selfplay_workers: int = 1  # §C.7 parallel self-play worker processes (1 = sequential); fills idle cores
    # §C.7 #3 SOLVER-FREE LEAGUE — break opening value-collapse WITHOUT an oracle (weak→strong non-oracle opponents +
    # seat-priority + a self-generated opening anchor). Reuses az_pool_frac as the league fraction. Default OFF.
    az_league: int = 0  # master switch; 1 = solver-free league (asserts az_distill_games==0 AND az_endgame_tablebase==0)
    az_league_p1_frac: float = 0.7  # P(learner on the won P1 seat) in league games — concentrates opening coverage
    az_league_snapshots: int = 4  # run-own arch-matched ckpt snapshots folded in as weak→recent AZ opponents
    az_league_anchor_frac: float = 0.25  # fixed-fraction pin for the self-generated opening anchor (0 = off)


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
    if not AZ_DISTILL_GAMES_RANGE[0] <= config.az_distill_games <= AZ_DISTILL_GAMES_RANGE[1]:
        raise ValueError(f"az_distill_games must be in {AZ_DISTILL_GAMES_RANGE}, got {config.az_distill_games}")
    if not AZ_SOLVE_ENDGAME_RANGE[0] <= config.az_solve_endgame <= AZ_SOLVE_ENDGAME_RANGE[1]:
        raise ValueError(f"az_solve_endgame must be in {AZ_SOLVE_ENDGAME_RANGE}, got {config.az_solve_endgame}")
    for name in ("az_endgame_tablebase", "az_endgame_exact_targets", "az_endgame_warm_start", "az_endgame_persist"):
        if getattr(config, name) not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1, got {getattr(config, name)}")
    if not AZ_ENDGAME_MAX_EMPTY_RANGE[0] <= config.az_endgame_max_empty <= AZ_ENDGAME_MAX_EMPTY_RANGE[1]:
        raise ValueError(f"az_endgame_max_empty must be in {AZ_ENDGAME_MAX_EMPTY_RANGE}, got {config.az_endgame_max_empty}")
    if not AZ_ENDGAME_EXTEND_POSITIONS_RANGE[0] <= config.az_endgame_extend_positions <= AZ_ENDGAME_EXTEND_POSITIONS_RANGE[1]:
        raise ValueError(f"az_endgame_extend_positions must be in {AZ_ENDGAME_EXTEND_POSITIONS_RANGE}, got {config.az_endgame_extend_positions}")
    if not AZ_ENDGAME_EXTEND_SECONDS_RANGE[0] <= config.az_endgame_extend_seconds <= AZ_ENDGAME_EXTEND_SECONDS_RANGE[1]:
        raise ValueError(f"az_endgame_extend_seconds must be in {AZ_ENDGAME_EXTEND_SECONDS_RANGE}, got {config.az_endgame_extend_seconds}")
    if not AZ_ENDGAME_CAP_RANGE[0] <= config.az_endgame_cap <= AZ_ENDGAME_CAP_RANGE[1]:
        raise ValueError(f"az_endgame_cap must be in {AZ_ENDGAME_CAP_RANGE}, got {config.az_endgame_cap}")
    for name in ("az_residual", "az_batchnorm"):
        if getattr(config, name) not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1, got {getattr(config, name)}")
    if not AZ_CHANNELS_RANGE[0] <= config.az_channels <= AZ_CHANNELS_RANGE[1]:
        raise ValueError(f"az_channels must be in {AZ_CHANNELS_RANGE}, got {config.az_channels}")
    if not AZ_BLOCKS_RANGE[0] <= config.az_blocks <= AZ_BLOCKS_RANGE[1]:
        raise ValueError(f"az_blocks must be in {AZ_BLOCKS_RANGE}, got {config.az_blocks}")
    if not AZ_HEAD_HIDDEN_RANGE[0] <= config.az_head_hidden <= AZ_HEAD_HIDDEN_RANGE[1]:
        raise ValueError(f"az_head_hidden must be in {AZ_HEAD_HIDDEN_RANGE}, got {config.az_head_hidden}")
    if not 1 <= config.az_selfplay_workers <= 32:
        raise ValueError(f"az_selfplay_workers must be in (1, 32), got {config.az_selfplay_workers}")
    if config.az_league not in (0, 1):
        raise ValueError(f"az_league must be 0 or 1, got {config.az_league}")
    if config.az_league and config.az_distill_games > 0:
        # league REPLACES oracle OPENING distillation (redundant + defeats the solver-free point). The grow-as-you-go
        # ENDGAME table (az_endgame_tablebase) is COMPLEMENTARY (endgame precision) and allowed alongside the league.
        raise ValueError("az_league=1 replaces oracle opening distillation: set az_distill_games=0 (endgame tablebase is allowed)")
    if not 0.0 <= config.az_league_p1_frac <= 1.0:
        raise ValueError(f"az_league_p1_frac must be in [0,1], got {config.az_league_p1_frac}")
    if not 0.0 <= config.az_league_anchor_frac < 1.0:
        raise ValueError(f"az_league_anchor_frac must be in [0,1), got {config.az_league_anchor_frac}")
    if not 0 <= config.az_league_snapshots <= 20:
        raise ValueError(f"az_league_snapshots must be in [0,20], got {config.az_league_snapshots}")


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
    for int_key in ("mcts_sims", "mcts_solve_endgame", "oracle_depth", "benchmark_positions", "eval_games", "seed", "az_iterations", "az_selfplay_games", "az_sims", "az_epochs", "az_distill_positions", "az_distill_games", "az_solve_endgame", "az_value_n_step", "az_target_refresh", "az_selfplay_opening_plies", "az_buffer_cap", "az_endgame_max_empty", "az_endgame_extend_positions", "az_endgame_cap", "az_channels", "az_blocks", "az_head_hidden", "az_selfplay_workers", "az_league_snapshots"):
        if int_key in filtered:
            filtered[int_key] = int(filtered[int_key])
    for float_key in ("az_c_scale", "az_pool_frac", "az_endgame_extend_seconds", "az_league_p1_frac", "az_league_anchor_frac"):
        if float_key in filtered:
            filtered[float_key] = float(filtered[float_key])
    for bool_key in ("az_warm_start", "az_gumbel", "az_endgame_tablebase", "az_endgame_exact_targets", "az_endgame_warm_start", "az_endgame_persist", "az_residual", "az_batchnorm", "az_league"):
        if bool_key in filtered:
            v = filtered[bool_key]
            filtered[bool_key] = 1 if (v is True or str(v).strip().lower() in ("1", "true", "yes")) else 0
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
