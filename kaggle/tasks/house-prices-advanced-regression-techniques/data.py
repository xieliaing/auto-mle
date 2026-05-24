"""
House Prices — Advanced Regression Techniques
AutoMLE data module (traditional ML mode).

load_train_dataset returns a DataFrame (80% split of train.csv).
load_eval_dataset returns a DataFrame (remaining 20%).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_TASK_DIR = Path(__file__).parent
_DATA_DIR = _TASK_DIR / "data"
_DEFAULT_TRAIN = _DATA_DIR / "train.csv"

TARGET = "SalePrice"


def _load_and_split(path: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * 0.8)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].reset_index(drop=True)


def load_train_dataset(
    seed: int = 42,
    subset: Optional[int] = None,
    balance: str = "none",
    **kwargs,
) -> pd.DataFrame:
    """Return training split (80% of train.csv) as a DataFrame."""
    path = Path(kwargs.get("train_path", _DEFAULT_TRAIN))
    train, _ = _load_and_split(path, seed)
    if subset and subset < len(train):
        train = train.sample(subset, random_state=seed).reset_index(drop=True)
    return train


def load_eval_dataset(
    seed: int = 42,
    subset: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Return eval split (20% of train.csv) as a DataFrame with SalePrice target."""
    path = Path(kwargs.get("eval_path", _DEFAULT_TRAIN))
    _, val = _load_and_split(path, seed)
    if subset and subset < len(val):
        val = val.sample(subset, random_state=seed).reset_index(drop=True)
    return val
