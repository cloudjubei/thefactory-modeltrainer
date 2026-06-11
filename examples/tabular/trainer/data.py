"""Data interface: declared dataset loading + the seeded train/val split.

The data contract: this program never downloads. The orchestrator materialises the
manifest's declared data before runs; a missing file is a hard, non-zero-exit error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/winequality-red.csv")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
TARGET_COLUMN = "quality"
VAL_FRACTION = 0.2


@dataclass(frozen=True)
class DatasetSplit:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series


def load_dataset_split(seed: int) -> DatasetSplit:
    if not DATA_FILE.is_file():
        raise SystemExit(
            f"{DATA_FILE} missing — launch via the Model Trainer app "
            f"(declared data materialises automatically) or download it from {DATA_URL}"
        )
    frame = pd.read_csv(DATA_FILE, sep=";")
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(float)
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, test_size=VAL_FRACTION, random_state=seed
    )
    return DatasetSplit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
