"""CLI entry: honors --config-json / --summary-out / --calibrate / --evaluate."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from trainer.config import TrainerConfig, config_hash, load_config, load_eval_config
from trainer.evaluate import evaluate_model
from trainer.model import build_model, build_train_env, load_model
from trainer.summary import build_eval_summary, build_summary, write_summary
from trainer.train import train_model

CALIBRATION_TIMESTEPS = 1000
CALIBRATION_EVAL_EPISODES = 5
EVAL_EPISODES = 20
CHECKPOINT_DIR = Path("checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(prog="trainer.run", description="CartPole trainer (model-training standard).")
    parser.add_argument("--config-json", type=Path, help="path to a fully resolved run config")
    parser.add_argument("--summary-out", type=Path, required=True, help="where to write the RunSummary JSON")
    parser.add_argument("--calibrate", action="store_true", help="tiny end-to-end pass reporting throughput")
    parser.add_argument("--evaluate", action="store_true", help="re-test a saved checkpoint without training")
    args = parser.parse_args()

    if args.evaluate and args.calibrate:
        parser.error("--evaluate and --calibrate are mutually exclusive")
    if args.config_json is None and not args.calibrate:
        parser.error("--config-json is required unless --calibrate is set")

    if args.evaluate:
        run_evaluation(args.config_json, args.summary_out)
    else:
        run_training(args.config_json, args.summary_out, args.calibrate)


def run_training(config_path: Path | None, summary_out: Path, calibrate: bool) -> None:
    if calibrate:
        config = replace(TrainerConfig(), algo="ppo", total_timesteps=CALIBRATION_TIMESTEPS)
        eval_episodes = CALIBRATION_EVAL_EPISODES
    else:
        config = load_config(config_path)
        eval_episodes = EVAL_EPISODES

    train_env = build_train_env()
    model = build_model(config, train_env)
    train_result = train_model(model, config.total_timesteps, train_env)
    eval_result = evaluate_model(model, seed=config.seed, episodes=eval_episodes)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{config_hash(config)}.zip"
    model.save(str(checkpoint))

    calibration = None
    if calibrate:
        calibration = {
            "unitsPerSecond": train_result.fps,
            "secondsObserved": train_result.seconds,
            "units": CALIBRATION_TIMESTEPS,
        }

    summary = build_summary(config, train_result, eval_result, str(checkpoint), calibration)
    write_summary(summary, summary_out)
    print(f"objective={summary['objective']:.1f} status={summary['health']['status']} summary={summary_out}")


def run_evaluation(config_path: Path, summary_out: Path) -> None:
    eval_config = load_eval_config(config_path)
    checkpoint = Path(eval_config.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    try:
        model = load_model(eval_config.config, checkpoint)
    except Exception as error:
        raise SystemExit(f"failed to load checkpoint {checkpoint}: {error}") from error

    eval_result = evaluate_model(model, seed=eval_config.config.seed, episodes=eval_config.eval_episodes)
    summary = build_eval_summary(eval_config.config, eval_result, eval_config.checkpoint)
    write_summary(summary, summary_out)
    print(f"objective={summary['objective']:.1f} status={summary['health']['status']} summary={summary_out}")


if __name__ == "__main__":
    main()
