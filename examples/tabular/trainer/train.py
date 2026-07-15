"""Training loop with wall-clock throughput and the per-stage validation curve.

Model-agnostic: the throughput count and the staged learning curve come from whatever API the chosen
regressor exposes (staged_predict for boosting; a single final point otherwise), so any `model_name` trains.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from sklearn.base import RegressorMixin

from trainer.data import DatasetSplit


@dataclass(frozen=True)
class TrainResult:
    seconds: float
    estimators_per_second: float
    val_rmse_per_stage: list[float]


def _n_estimators(model: RegressorMixin) -> int:
    # Fitted ensemble size across regressor families (GB: n_estimators_, forests: len(estimators_),
    # HistGradientBoosting: n_iter_), falling back to the requested count.
    for attr in ("n_estimators_", "n_iter_"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        return len(estimators)
    return int(getattr(model, "n_estimators", 1) or 1)


def train_model(model: RegressorMixin, split: DatasetSplit) -> TrainResult:
    start = time.perf_counter()
    model.fit(split.x_train, split.y_train)
    seconds = time.perf_counter() - start
    return TrainResult(
        seconds=seconds,
        estimators_per_second=_n_estimators(model) / seconds if seconds > 0 else 0.0,
        val_rmse_per_stage=staged_val_rmse(model, split),
    )


def staged_val_rmse(model: RegressorMixin, split: DatasetSplit) -> list[float]:
    y_val = split.y_val.to_numpy()
    # Boosting models expose a per-stage curve; others only a final fit — report the single end-point RMSE.
    if hasattr(model, "staged_predict"):
        return [
            math.sqrt(float(((predictions - y_val) ** 2).mean()))
            for predictions in model.staged_predict(split.x_val)
        ]
    final = model.predict(split.x_val)
    return [math.sqrt(float(((final - y_val) ** 2).mean()))]
