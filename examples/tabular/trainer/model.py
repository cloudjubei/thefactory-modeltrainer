"""Model factory: sklearn GradientBoostingRegressor + joblib checkpoint IO."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor

from trainer.config import TrainerConfig


def build_model(config: TrainerConfig) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        random_state=config.seed,
    )


def save_model(model: GradientBoostingRegressor, checkpoint: Path) -> None:
    joblib.dump(model, checkpoint)


def load_model(checkpoint: Path) -> GradientBoostingRegressor:
    return joblib.load(checkpoint)
