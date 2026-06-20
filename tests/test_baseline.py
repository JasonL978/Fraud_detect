"""Synthetic end-to-end tests for the baseline feature build + training."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_detect import constants as C
from fraud_detect.baseline import BaselineParams, train_and_evaluate
from fraud_detect.baseline_features import build_baseline_features, time_split_masks
from fraud_detect.synth import make_synthetic_paysim


def test_build_features_shape_and_columns():
    df = make_synthetic_paysim(100)
    X, y = build_baseline_features(df)
    assert len(X) == len(y) == 100
    for t in C.TX_TYPES:
        assert f"type_{t}" in X.columns
    assert "errorBalanceOrig" in X.columns and "errorBalanceDest" in X.columns
    assert not X.isnull().any().any()


def test_build_features_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        build_baseline_features(pd.DataFrame({"step": [1], "type": ["TRANSFER"]}))


def test_time_split_masks_disjoint_and_drop_embargo():
    df = make_synthetic_paysim(2000)
    train_mask, test_mask = time_split_masks(df)
    # No row is in both splits.
    assert not (train_mask & test_mask).any()
    # Embargo rows are in neither.
    embargo = (df["step"].to_numpy() > C.TRAIN_END_STEP) & \
              (df["step"].to_numpy() < C.TEST_START_STEP)
    assert not (train_mask & embargo).any()
    assert not (test_mask & embargo).any()


def test_train_and_evaluate_learns_signal():
    df = make_synthetic_paysim(4000)
    result = train_and_evaluate(df, BaselineParams(n_estimators=50, max_depth=4))
    m = result.metrics
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["recall_at_1pct_fpr"] <= 1.0
    # The signal is strong and learnable; a real model should beat the base rate.
    assert m["pr_auc"] > m["base_rate"]
    assert result.scale_pos_weight > 1.0  # imbalanced -> upweight positives
    assert result.split_sizes["train"] > 0 and result.split_sizes["test"] > 0
    assert result.precision is not None and result.recall is not None
