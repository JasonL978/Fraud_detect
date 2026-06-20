"""Synthetic PaySim-like data, for tests and offline pipeline validation.

NOT a fraud model evaluation tool — this exists so the pipeline can be exercised
without the real PaySim download. Numbers produced from this data are
placeholders, never reported as results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C


def make_synthetic_paysim(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthetic PaySim-like frame spanning train/embargo/test steps, with a
    learnable signal (fraud drains the origin account and mismatches the dest)."""
    rng = np.random.default_rng(seed)
    step = rng.integers(C.MIN_STEP, C.MAX_STEP + 1, size=n)
    amount = rng.uniform(1, 5000, size=n)
    old_org = amount + rng.uniform(0, 5000, size=n)
    is_fraud = rng.random(size=n) < 0.05

    new_org = np.where(is_fraud, 0.0, old_org - amount)
    old_dest = rng.uniform(0, 5000, size=n)
    new_dest = np.where(is_fraud, old_dest, old_dest + amount)

    types = rng.choice(C.TX_TYPES, size=n)
    return pd.DataFrame({
        "step": step,
        "type": types,
        "amount": amount,
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "nameDest": [f"C{i + n}" for i in range(n)],
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud.astype(int),
        "isFlaggedFraud": 0,
    })
