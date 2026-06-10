"""CLI entry: honors --config-json / --summary-out / --calibrate."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from trainer.config import TrainerConfig, config_hash, load_config
from trainer.evaluate import evaluate_model
from trainer.model import build_model
from trainer.summary import build_summary, write_summary
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
    args = parser.parse_args()

    if args.calibrate:
        config = replace(TrainerConfig(), algo="ppo", total_timesteps=CALIBRATION_TIMESTEPS)
        eval_episodes = CALIBRATION_EVAL_EPISODES
    else:
        if args.config_json is None:
            parser.error("--config-json is required unless --calibrate is set")
        config = load_config(args.config_json)
        eval_episodes = EVAL_EPISODES

    model = build_model(config)
    train_result = train_model(model, config.total_timesteps)
    eval_result = evaluate_model(model, seed=config.seed, episodes=eval_episodes)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{config_hash(config)}.zip"
    model.save(str(checkpoint))

    calibration = None
    if args.calibrate:
        calibration = {
            "unitsPerSecond": train_result.fps,
            "secondsObserved": train_result.seconds,
            "units": CALIBRATION_TIMESTEPS,
        }

    summary = build_summary(config, train_result, eval_result, str(checkpoint), calibration)
    write_summary(summary, args.summary_out)
    print(f"objective={summary['objective']:.1f} status={summary['health']['status']} summary={args.summary_out}")


if __name__ == "__main__":
    main()
