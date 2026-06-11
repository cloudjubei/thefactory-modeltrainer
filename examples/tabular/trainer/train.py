"""Training loop with wall-clock throughput and the per-stage validation curve."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor

from trainer.data import DatasetSplit


@dataclass(frozen=True)
class TrainResult:
    seconds: float
    estimators_per_second: float
    val_rmse_per_stage: list[float]


def train_model(model: GradientBoostingRegressor, split: DatasetSplit) -> TrainResult:
    start = time.perf_counter()
    model.fit(split.x_train, split.y_train)
    seconds = time.perf_counter() - start
    return TrainResult(
        seconds=seconds,
        estimators_per_second=model.n_estimators_ / seconds,
        val_rmse_per_stage=staged_val_rmse(model, split),
    )


def staged_val_rmse(model: GradientBoostingRegressor, split: DatasetSplit) -> list[float]:
    y_val = split.y_val.to_numpy()
    return [
        math.sqrt(float(((predictions - y_val) ** 2).mean()))
        for predictions in model.staged_predict(split.x_val)
    ]
