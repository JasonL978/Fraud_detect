"""Map a raw PaySim row to the target ``transactions.v1`` event shape.

Design notes (see the design doc Week 2 deep dive):

* PaySim's columns are NOT the topic schema. This module is the mapping layer
  so PaySim's shape never leaks downstream.
* ``event_id`` and the intra-hour ``jitter`` are DETERMINISTIC functions of the
  source row's stable identity (its CSV row index). This is mandatory for
  online/offline parity: an offline recompute must reproduce the exact same
  timestamp, or the parity test can never pass. We use ``hashlib`` (stable
  across processes/platforms), never the builtin ``hash`` (per-process salted).
* ``step`` (an hour bucket) is spread across its hour by the seeded jitter so
  that sub-hour velocity windows (10s/30s/1m) are not degenerate, without
  introducing artificial periodicity.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import fastavro

from . import constants as C

# Resolve data/schemas/transaction_v1.avsc from this file's location.
# src/fraud_detect/paysim_mapping.py -> parents[2] == repo root.
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schemas" / "transaction_v1.avsc"

# Stable UUIDv5 namespace for deterministic event_ids.
_EVENT_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "transactions.v1.fraud-detect")


class UnknownTxTypeError(ValueError):
    """Raised when a PaySim `type` is not in the frozen TxType enum.

    Day 2 fails loudly; Day 4 will route these to a review bucket (RT-014)
    instead of raising.
    """


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """Parse and cache the Avro schema for transactions.v1."""
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return fastavro.parse_schema(json.load(fh))


def _stable_hash_int(row_index: int) -> int:
    """Deterministic, cross-platform hash of a row key -> non-negative int."""
    digest = hashlib.sha256(f"{C.HASH_SEED}:{row_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def compute_jitter_ms(row_index: int) -> int:
    """Deterministic intra-step offset in [0, STEP_DURATION_MS)."""
    return _stable_hash_int(row_index) % C.STEP_DURATION_MS


def make_event_id(row_index: int) -> str:
    """Deterministic UUIDv5 event id from the stable row identity."""
    return str(uuid.uuid5(_EVENT_NS, str(row_index)))


def step_to_event_time(step: int, row_index: int) -> datetime:
    """PaySim `step` (1-based hour index) -> UTC, jittered within the hour."""
    base = C.EPOCH + timedelta(hours=step - 1)
    return base + timedelta(milliseconds=compute_jitter_ms(row_index))


def map_paysim_row(row: Mapping[str, Any], row_index: int) -> dict:
    """Map one PaySim row (dict-like) to a transactions.v1 record.

    ``row_index`` is the row's stable position in the source CSV; it seeds the
    deterministic event_id and jitter, so the same input always yields the same
    output (parity requirement).

    Tier-2 fields (device/funding/geo/channel/recipient_age) are left null —
    they are injected by later generators. Computed features never appear here.
    """
    tx_type = str(row["type"]).strip().upper()
    if tx_type not in C.TX_TYPES:
        raise UnknownTxTypeError(
            f"row {row_index}: unknown tx_type {tx_type!r}; "
            f"expected one of {C.TX_TYPES}"
        )

    step = int(row["step"])

    return {
        "event_id": make_event_id(row_index),
        "event_timestamp": step_to_event_time(step, row_index),
        "tx_type": tx_type,
        "amount": float(row["amount"]),
        "sender_id": str(row["nameOrig"]),
        "recipient_id": str(row["nameDest"]),
        "sender_old_balance": float(row["oldbalanceOrg"]),
        "sender_new_balance": float(row["newbalanceOrig"]),
        "recipient_old_balance": float(row["oldbalanceDest"]),
        "recipient_new_balance": float(row["newbalanceDest"]),
        "source_step": step,
        "label_is_fraud": bool(int(row["isFraud"])),
        # Tier-2: injected later.
        "funding_source": None,
        "device_id": None,
        "recipient_account_age_days": None,
        "geo_region": None,
        "channel": None,
    }


def validate_record(record: Mapping[str, Any]) -> bool:
    """Validate a mapped record against the Avro schema. Raises on failure."""
    return fastavro.validation.validate(dict(record), load_schema(), raise_errors=True)
