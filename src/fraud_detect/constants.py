"""Frozen project constants.

These define the synthetic timeline and the train/test split. They are
LOAD-BEARING for correctness and must not change once data/numbers are
committed: the time-ordered split, point-in-time parity joins, and the canary
set all anchor to them. Changing EPOCH or the split boundaries invalidates every
committed metric and the canary hash. Treat edits here like a data migration.
"""

from __future__ import annotations

from datetime import datetime, timezone

# --- Synthetic timeline -----------------------------------------------------

# Arbitrary but FROZEN base time. PaySim `step` is an hour index (1..744);
# we anchor step 1 to this instant. Never change.
EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Each PaySim step spans one hour.
STEP_DURATION_MS = 3_600_000

# PaySim spans 30 days = 744 hourly steps.
MIN_STEP = 1
MAX_STEP = 744

# --- Time-ordered train / test split ---------------------------------------
#
# Split is BY TIME (step), never random — a random split leaks future fraud
# into training and massively inflates metrics at 0.13% prevalence.
#
# An EMBARGO gap sits between train and test, at least as wide as the longest
# rolling feature window, so a test-set feature window cannot reach back into
# training data (leakage *through* the feature window). With the v1 cap of a
# 24h max window, a 24-step embargo suffices.
#
#   train:   steps [1, 600]
#   embargo: steps [601, 624]   (dropped from both sets)
#   test:    steps [625, 744]
TRAIN_END_STEP = 600
MAX_FEATURE_WINDOW_HOURS = 24          # v1 cap; revisit before adding 7d windows
EMBARGO_STEPS = MAX_FEATURE_WINDOW_HOURS
TEST_START_STEP = TRAIN_END_STEP + EMBARGO_STEPS + 1   # 625

# Deterministic-hash namespace seed. Bump only if you intend to re-roll all
# event_ids and jitter (which re-rolls the whole synthetic timeline).
HASH_SEED = "fraud-detect:v1"

# Frozen 5-symbol payment enum (mirrors transaction_v1.avsc TxType).
TX_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

# Feature windows for streaming/offline aggregation, ordered shortest -> longest.
# Capped at 24h to match the embargo (MAX_FEATURE_WINDOW_HOURS); no 7d in v1.
# (label, milliseconds)
FEATURE_WINDOWS = (
    ("10s", 10_000),
    ("30s", 30_000),
    ("1m", 60_000),
    ("5m", 300_000),
    ("1h", 3_600_000),
    ("24h", 86_400_000),
)
MAX_FEATURE_WINDOW_MS = FEATURE_WINDOWS[-1][1]

EPOCH_MS = int(EPOCH.timestamp() * 1000)


def time_split_for_step(step: int) -> str:
    """Return 'train', 'embargo', or 'test' for a PaySim step.

    'embargo' rows are dropped from both training and evaluation.
    """
    if step <= TRAIN_END_STEP:
        return "train"
    if step < TEST_START_STEP:
        return "embargo"
    return "test"
