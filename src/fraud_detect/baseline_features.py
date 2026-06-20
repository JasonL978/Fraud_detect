"""Vectorized baseline features from raw PaySim, plus the time-ordered split.

These are computed directly on the raw PaySim DataFrame (not via the per-row
event mapper) because the baseline trains on millions of static rows. All
features are same-row functions of the transaction, so there is no temporal
leakage; leakage is prevented by the time split, not the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C

# Raw PaySim columns the baseline needs.
_REQUIRED = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud",
]

# Numeric features carried straight through.
_NUMERIC = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def build_baseline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the XGBoost baseline.

    Features: raw amount + balances, two balance-error signals (PaySim's known
    strong feature), and one-hot ``type`` over the frozen 5-symbol vocabulary.
    """
    missing = set(_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"PaySim frame missing columns: {sorted(missing)}")

    feats = df[_NUMERIC].astype("float64").copy()

    # Balance-error: for a consistent ledger, new = old - amount (orig) and
    # new = old + amount (dest). Non-zero error is a strong fraud signal.
    feats["errorBalanceOrig"] = (
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    ).astype("float64")
    feats["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).astype("float64")

    # One-hot type over the FROZEN vocabulary, so columns are stable regardless
    # of which types appear in a given slice of data.
    type_upper = df["type"].astype("string").str.upper()
    for t in C.TX_TYPES:
        feats[f"type_{t}"] = (type_upper == t).astype("int8")

    y = df["isFraud"].astype("int8")
    return feats, y


def time_split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, test_mask) by PaySim step; embargo rows are in neither."""
    step = df["step"].to_numpy()
    train_mask = step <= C.TRAIN_END_STEP
    test_mask = step >= C.TEST_START_STEP
    return train_mask, test_mask
