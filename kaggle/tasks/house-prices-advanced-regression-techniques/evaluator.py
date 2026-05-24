"""
House Prices evaluator — traditional ML mode.

Contract:
  PRIMARY_METRIC = "rmsle"
  evaluate(adapter_dir, model_name, eval_data, **kwargs) -> dict

Traditional ML path (used by kaggle/loop.py):
  kwargs["train_data"]   — training DataFrame; triggers fit + save
  kwargs["config"]       — ExperimentConfig dict; extracts n_estimators / lr
  kwargs["output_dir"]   — where to save model.pkl
  kwargs["predictions_path"] — where to write predictions.csv

predictions_df columns (binary encoding for defect analysis):
  sample_id   — row index in eval_data
  true_label  — 1 if log1p(SalePrice) >= LOG_MEDIAN, else 0  (above/below median)
  pred_label  — 1 if log1p(y_pred) >= LOG_MEDIAN, else 0
  confidence  — sigmoid((log1p(y_pred) - LOG_MEDIAN) * 2) ∈ (0, 1)
              — near 0.5 = uncertain (near boundary); near 0/1 = confident

Note: LOG_MEDIAN is fixed at log1p(163000) ≈ 12.0 — the approximate median
SalePrice for the Ames dataset.  Kept stable across iterations so defect
comparisons are consistent.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET = "SalePrice"

# Fixed threshold: log1p(163,000) — Ames housing dataset median SalePrice.
# Kept constant across all iterations so defect labels are comparable.
LOG_MEDIAN = np.log1p(163_000)

PRIMARY_METRIC = "rmsle"


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def _build_pipeline(n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 4) -> Pipeline:
    cat_imp = SimpleImputer(strategy="most_frequent")
    num_imp = SimpleImputer(strategy="median")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # handles mixed after encoding
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            subsample=0.8,
        )),
    ])


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID and target; encode categoricals as ordinal integers."""
    drop_cols = [c for c in ("Id", TARGET) if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes  # -1 for NaN
    return X


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_predictions_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids,
) -> pd.DataFrame:
    """
    Build predictions DataFrame for defect analysis.
    Binary encoding: label=1 if log1p(price) >= LOG_MEDIAN (above Ames median).
    Confidence = sigmoid((log1p(y_pred) - LOG_MEDIAN) * 2).
    """
    log_true = np.log1p(y_true)
    log_pred = np.log1p(np.maximum(y_pred, 0))

    true_label = (log_true >= LOG_MEDIAN).astype(int)
    pred_label = (log_pred >= LOG_MEDIAN).astype(int)
    confidence = _sigmoid((log_pred - LOG_MEDIAN) * 2)

    return pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": np.round(confidence, 6),
        "true_sale_price": y_true.astype(int),
        "pred_sale_price": np.maximum(y_pred, 0).astype(int),
        "log_residual": np.round(np.abs(log_pred - log_true), 6),
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    adapter_dir: Optional[str],
    model_name: str,
    eval_data: pd.DataFrame,
    **kwargs,
) -> dict:
    """
    Train (if train_data provided) and evaluate on eval_data.

    Returns RMSLE, n_eval, predictions_df.
    """
    output_dir = Path(kwargs.get("output_dir", adapter_dir or "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    predictions_path = kwargs.get("predictions_path")

    # ---- Extract model params from ExperimentConfig dict -------------------
    config = kwargs.get("config", {})
    train_cfg = config.get("train", {})
    n_estimators = max(50, int(round(train_cfg.get("num_epochs", 1.0) * 200)))
    learning_rate = float(train_cfg.get("learning_rate", 0.05))

    # ---- Train if training data provided -----------------------------------
    train_data = kwargs.get("train_data")
    if train_data is not None:
        X_train = _prepare_X(train_data)
        y_train = train_data[TARGET].values
        pipe = _build_pipeline(n_estimators=n_estimators, learning_rate=learning_rate)
        pipe.fit(X_train, y_train)
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
    else:
        with open(model_path, "rb") as f:
            pipe = pickle.load(f)

    # ---- Evaluate ----------------------------------------------------------
    X_eval = _prepare_X(eval_data)
    y_true = eval_data[TARGET].values
    y_pred = pipe.predict(X_eval)

    score = _rmsle(y_true, y_pred)

    predictions_df = _make_predictions_df(
        y_true, y_pred,
        sample_ids=eval_data.index.tolist(),
    )

    if predictions_path:
        predictions_df.to_csv(predictions_path, index=False)

    return {
        PRIMARY_METRIC: score,
        "n_eval": len(eval_data),
        "predictions_df": predictions_df,
    }
