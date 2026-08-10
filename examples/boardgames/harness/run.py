"""CLI entry: honors --config-json / --summary-out / --calibrate / --evaluate (the model-training standard).

A run EVALUATES `model_name` against one `opponent` rung over `eval_games` games and writes a §C RunSummary.
The saved checkpoint is a small JSON spec a `load_policy` seam turns back into a playable agent — the artifact
you take to a live test (e.g. a BoardGameArena bridge).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from harness.agents import Agent, resolve_agent
from harness.config import TrainerConfig, config_hash, load_config, load_eval_config
from harness.game import Game, State
from harness.registry import personas_for, resolve_game
from harness.selfplay import evaluate_vs_opponent
from harness.summary import build_eval_summary, build_summary, write_summary

CALIBRATION_GAMES = 6
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.run", description="Board-game trainer (model-training standard).")
    parser.add_argument("--config-json", type=Path, help="path to a fully resolved run config")
    parser.add_argument("--summary-out", type=Path, required=True, help="where to write the RunSummary JSON")
    parser.add_argument("--calibrate", action="store_true", help="tiny end-to-end pass reporting throughput")
    parser.add_argument("--evaluate", action="store_true", help="re-test a saved checkpoint")
    args = parser.parse_args()

    if args.evaluate and args.calibrate:
        parser.error("--evaluate and --calibrate are mutually exclusive")
    if args.config_json is None and not args.calibrate:
        parser.error("--config-json is required unless --calibrate is set")

    if args.evaluate:
        _run_evaluation(args.config_json, args.summary_out)
    else:
        _run_training(args.config_json, args.summary_out, args.calibrate)


def _run_training(config_path: Path | None, summary_out: Path, calibrate: bool) -> None:
    if calibrate:
        config = replace(TrainerConfig(), model_name="mcts", mcts_sims=40, opponent="random", eval_games=CALIBRATION_GAMES)
    else:
        config = load_config(config_path)

    game = resolve_game(config.game)
    rng = random.Random(config.seed)

    wall0, cpu0 = time.perf_counter(), time.process_time()
    ev = evaluate_vs_opponent(
        game, config.model_name, config.opponent, asdict(config), config.eval_games, rng, personas_for(config.game)
    )
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    steps = {"selfplay_seconds": wall}

    checkpoint = _save_checkpoint(config)
    calibration = None
    if calibrate:
        calibration = {"unitsPerSecond": (config.eval_games / wall) if wall > 0 else 0.0, "secondsObserved": wall, "units": config.eval_games}

    summary = build_summary(config, ev, checkpoint, wall, cpu, steps, calibration)
    write_summary(summary, summary_out)
    print(f"win_rate={ev.win_rate:.3f} vs {config.opponent} cost=${summary['cost']['estCostUsd']:.6f} summary={summary_out}")


def _run_evaluation(config_path: Path, summary_out: Path) -> None:
    eval_config = load_eval_config(config_path)
    checkpoint = Path(eval_config.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    spec = json.loads(checkpoint.read_text())
    config = replace(load_checkpoint_config(spec), opponent=eval_config.config.opponent, eval_games=eval_config.eval_games, seed=eval_config.config.seed)

    game = resolve_game(config.game)
    rng = random.Random(config.seed)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    ev = evaluate_vs_opponent(game, config.model_name, config.opponent, asdict(config), config.eval_games, rng, personas_for(config.game))
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0

    summary = build_eval_summary(config, ev, str(checkpoint), wall, cpu, {"eval_seconds": wall})
    write_summary(summary, summary_out)
    print(f"win_rate={ev.win_rate:.3f} vs {config.opponent} (eval of {checkpoint.name}) summary={summary_out}")


def _save_checkpoint(config: TrainerConfig) -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{config_hash(config)}.json"
    # The playable spec: which game + core + strength. A neural core would save weights alongside this.
    path.write_text(json.dumps({"game": config.game, "model_name": config.model_name, "mcts_sims": config.mcts_sims}, indent=2) + "\n")
    return str(path)


def load_checkpoint_config(spec: dict) -> TrainerConfig:
    return TrainerConfig(game=spec["game"], model_name=spec["model_name"], mcts_sims=int(spec.get("mcts_sims", 80)))


def load_policy(checkpoint_path: str | Path) -> Callable[[Game, State, random.Random], int]:
    """Reconstruct a playable agent from a checkpoint — the seam a live test / BGA bridge drives.

    Returns `act(game, state, rng) -> action`. A search core needs the full state; a learned core would act
    from `game.observation(state, player)` alone (what a live venue exposes).
    """
    spec = json.loads(Path(checkpoint_path).read_text())
    config = load_checkpoint_config(spec)

    def act(game: Game, state: State, rng: random.Random) -> int:
        agent: Agent = resolve_agent(config.model_name, asdict(config), personas_for(config.game))
        return agent.act(game, state, rng)

    return act


if __name__ == "__main__":
    main()
