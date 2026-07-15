"""Held-out evaluation: seeded validation-split metrics + health flags."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

from trainer.data import DatasetSplit


@dataclass(frozen=True)
class EvalResult:
    val_rmse: float
    train_rmse: float
    val_r2: float
    n_train: int
    n_val: int
    flags: list[str] = field(default_factory=list)


def evaluate_model(model: RegressorMixin, split: DatasetSplit) -> EvalResult:
    val_rmse = math.sqrt(float(mean_squared_error(split.y_val, model.predict(split.x_val))))
    train_rmse = math.sqrt(float(mean_squared_error(split.y_train, model.predict(split.x_train))))
    val_r2 = float(r2_score(split.y_val, model.predict(split.x_val)))
    flags: list[str] = []
    if any(math.isnan(value) for value in (val_rmse, train_rmse, val_r2)):
        flags.append("nan_metrics")
    return EvalResult(
        val_rmse=val_rmse,
        train_rmse=train_rmse,
        val_r2=val_r2,
        n_train=len(split.y_train),
        n_val=len(split.y_val),
        flags=flags,
    )
