"""Tests for the PaySim -> transactions.v1 mapping.

Focus is the parity-critical invariants: determinism of event_id/timestamp,
jitter spread within the hour, schema validity, and the time-split boundaries.
"""

from __future__ import annotations

import io
from datetime import timedelta, timezone

import pytest
from fastavro import schemaless_reader, schemaless_writer

from fraud_detect import constants as C
from fraud_detect.paysim_mapping import (
    UnknownTxTypeError,
    compute_jitter_ms,
    load_schema,
    make_event_id,
    map_paysim_row,
    step_to_event_time,
    validate_record,
)


def sample_row(**overrides):
    row = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C1666544295",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 1,
        "isFlaggedFraud": 0,
    }
    row.update(overrides)
    return row


def test_mapped_record_validates_against_schema():
    rec = map_paysim_row(sample_row(), row_index=0)
    assert validate_record(rec) is True


def test_avro_binary_roundtrip_preserves_values():
    """Encode -> decode through Avro binary; the timestamp logical type is the
    risky part. Catches encoding bugs without needing Kafka/Docker."""
    rec = map_paysim_row(sample_row(), row_index=5)
    schema = load_schema()

    buf = io.BytesIO()
    schemaless_writer(buf, schema, rec)
    buf.seek(0)
    out = schemaless_reader(buf, schema)

    # timestamp-millis survives at ms precision and stays tz-aware UTC.
    assert out["event_timestamp"] == rec["event_timestamp"]
    assert out["event_timestamp"].tzinfo is not None
    assert out["event_timestamp"].utcoffset() == timezone.utc.utcoffset(None)
    assert out["tx_type"] == "TRANSFER"
    assert out["amount"] == rec["amount"]
    assert out["source_step"] == 1
    assert out["label_is_fraud"] is True
    # Tier-2 nullable fields round-trip as None.
    assert out["funding_source"] is None
    assert out["device_id"] is None


def test_mapping_is_deterministic():
    a = map_paysim_row(sample_row(), row_index=42)
    b = map_paysim_row(sample_row(), row_index=42)
    assert a == b
    assert a["event_id"] == b["event_id"]
    assert a["event_timestamp"] == b["event_timestamp"]


def test_event_id_and_jitter_differ_by_row_index():
    assert make_event_id(1) != make_event_id(2)
    # Jitter should not be constant across rows (would collapse sub-hour windows).
    jitters = {compute_jitter_ms(i) for i in range(50)}
    assert len(jitters) > 40


def test_jitter_within_hour_bounds():
    for i in range(1000):
        j = compute_jitter_ms(i)
        assert 0 <= j < C.STEP_DURATION_MS


def test_timestamp_anchors_to_epoch_and_jitter():
    rec = map_paysim_row(sample_row(step=1), row_index=7)
    expected = C.EPOCH + timedelta(milliseconds=compute_jitter_ms(7))
    assert rec["event_timestamp"] == expected


def test_step_advances_one_hour():
    t1 = step_to_event_time(step=1, row_index=7)
    t2 = step_to_event_time(step=2, row_index=7)
    assert (t2 - t1) == timedelta(hours=1)  # same row_index -> same jitter cancels


def test_tier2_fields_are_null():
    rec = map_paysim_row(sample_row(), row_index=0)
    for f in ("funding_source", "device_id", "recipient_account_age_days",
              "geo_region", "channel"):
        assert rec[f] is None


def test_label_and_type_mapping():
    fraud = map_paysim_row(sample_row(isFraud=1, type="cash_out"), row_index=0)
    assert fraud["label_is_fraud"] is True
    assert fraud["tx_type"] == "CASH_OUT"  # normalized upper-case
    benign = map_paysim_row(sample_row(isFraud=0), row_index=0)
    assert benign["label_is_fraud"] is False


def test_unknown_tx_type_raises():
    with pytest.raises(UnknownTxTypeError):
        map_paysim_row(sample_row(type="CRYPTO"), row_index=0)


@pytest.mark.parametrize(
    "step,expected",
    [
        (1, "train"),
        (C.TRAIN_END_STEP, "train"),
        (C.TRAIN_END_STEP + 1, "embargo"),
        (C.TEST_START_STEP - 1, "embargo"),
        (C.TEST_START_STEP, "test"),
        (C.MAX_STEP, "test"),
    ],
)
def test_time_split_boundaries(step, expected):
    assert C.time_split_for_step(step) == expected


def test_embargo_is_at_least_max_feature_window():
    # The embargo gap must cover the longest rolling window to prevent leakage
    # through the feature window.
    assert C.EMBARGO_STEPS >= C.MAX_FEATURE_WINDOW_HOURS
