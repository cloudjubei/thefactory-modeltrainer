"""Model factory: sklearn regressors selected by `model_name` + joblib checkpoint IO.

The `model_name` lever is the model-identity axis the project scan discovers and the exploration autopilot
treats as a basin axis (which model wins), while the ensemble hyperparameters tune WITHIN a model. Each
regressor consumes the hyperparameters that are meaningful to it (e.g. random_forest has no learning_rate),
so an inapplicable lever simply has no effect for that model — the conditional-importance the tools expect.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from trainer.config import TrainerConfig


def _max_samples(subsample: float) -> float | None:
    # sklearn bagging draws a fraction in (0, 1]; a full sample is expressed as None (use every row).
    return subsample if subsample < 1.0 else None


def build_model(config: TrainerConfig) -> RegressorMixin:
    name = config.model_name
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=config.subsample,
            random_state=config.seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=config.seed,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            max_samples=_max_samples(config.subsample),
            bootstrap=True,
            random_state=config.seed,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            max_samples=_max_samples(config.subsample),
            bootstrap=True,
            random_state=config.seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model_name: {name!r}")


def save_model(model: RegressorMixin, checkpoint: Path) -> None:
    joblib.dump(model, checkpoint)


def load_model(checkpoint: Path) -> RegressorMixin:
    return joblib.load(checkpoint)
