"""Train + evaluate the XGBoost baseline. No file I/O, no MLflow (testable core)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .baseline_features import build_baseline_features, time_split_masks
from .metrics import evaluate


@dataclass
class BaselineParams:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


@dataclass
class BaselineResult:
    model: XGBClassifier
    metrics: dict
    feature_columns: list[str]
    scale_pos_weight: float
    split_sizes: dict
    # PR-curve points for plotting downstream.
    precision: np.ndarray = field(repr=False, default=None)
    recall: np.ndarray = field(repr=False, default=None)


def _scale_pos_weight(y: pd.Series) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return (neg / pos) if pos else 1.0


def train_and_evaluate(df: pd.DataFrame, params: BaselineParams | None = None) -> BaselineResult:
    """Time-split, train XGBoost with scale_pos_weight, evaluate on the test split."""
    params = params or BaselineParams()
    from sklearn.metrics import precision_recall_curve

    X, y = build_baseline_features(df)
    train_mask, test_mask = time_split_masks(df)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("empty train or test split; check step ranges in the data")

    spw = _scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        random_state=params.random_state,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), scores)
    precision, recall, _ = precision_recall_curve(y_test.to_numpy(), scores)

    return BaselineResult(
        model=model,
        metrics=metrics,
        feature_columns=list(X.columns),
        scale_pos_weight=spw,
        split_sizes={"train": int(len(X_train)), "test": int(len(X_test))},
        precision=precision,
        recall=recall,
    )
