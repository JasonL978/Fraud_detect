"""Tests for the RT-014 ingest-path schema validator."""

from __future__ import annotations

import pytest

from fraud_detect import constants as C
from fraud_detect.paysim_mapping import map_paysim_row
from fraud_detect.schema_validator import (
    ReviewRouter,
    enum_vocabulary,
    required_fields,
    validate_event,
)


def _good_record():
    row = {
        "step": 1, "type": "TRANSFER", "amount": 181.0,
        "nameOrig": "C1", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
        "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "isFraud": 1, "isFlaggedFraud": 0,
    }
    return map_paysim_row(row, row_index=0)


# --- vocabulary derivation ---------------------------------------------------

def test_vocabulary_matches_constants():
    """Schema-derived tx_type symbols must match the hardcoded constant."""
    assert enum_vocabulary()["tx_type"] == frozenset(C.TX_TYPES)


def test_enum_fields_discovered():
    assert set(enum_vocabulary()) == {"tx_type", "funding_source", "channel"}


def test_label_excluded_from_required():
    assert "label_is_fraud" not in required_fields()
    assert "tx_type" in required_fields()
    assert "amount" in required_fields()


# --- validate_event ----------------------------------------------------------

def test_good_record_accepted():
    assert validate_event(_good_record()).accepted


def test_unknown_tx_type_routed_to_review():
    rec = dict(_good_record(), tx_type="CRYPTO")
    result = validate_event(rec)
    assert not result.accepted
    assert "unknown_enum:tx_type=CRYPTO" in result.reasons


def test_unknown_nullable_enum_routed_to_review():
    rec = dict(_good_record(), channel="CARRIER_PIGEON")
    result = validate_event(rec)
    assert not result.accepted
    assert "unknown_enum:channel=CARRIER_PIGEON" in result.reasons


def test_null_nullable_enum_is_fine():
    rec = dict(_good_record(), channel=None, funding_source=None)
    assert validate_event(rec).accepted


def test_missing_required_field_routed_to_review():
    rec = _good_record()
    del rec["amount"]
    result = validate_event(rec)
    assert not result.accepted
    assert "missing_field:amount" in result.reasons


def test_null_required_field_routed_to_review():
    rec = dict(_good_record(), sender_id=None)
    result = validate_event(rec)
    assert not result.accepted
    assert "missing_field:sender_id" in result.reasons


def test_missing_label_does_not_trigger_review():
    """An inference-path event without the oracle label must still be accepted."""
    rec = _good_record()
    del rec["label_is_fraud"]
    assert validate_event(rec).accepted


@pytest.mark.parametrize("bad", [None, [1, 2, 3], "not-a-record", 42])
def test_malformed_payload_routed_to_review_not_raised(bad):
    """A non-mapping payload must be diverted to review, never crash the loop."""
    result = validate_event(bad)
    assert not result.accepted
    assert result.reasons == ["malformed:not_a_mapping"]


def test_router_handles_malformed_without_raising():
    review = []
    router = ReviewRouter(on_accept=lambda r: None,
                          on_review=lambda r, reasons: review.append(reasons))
    router.route(None)
    assert review == [["malformed:not_a_mapping"]]


def test_multiple_reasons_collected():
    rec = dict(_good_record(), tx_type="CRYPTO", channel="CARRIER_PIGEON")
    result = validate_event(rec)
    assert len(result.reasons) == 2


# --- ReviewRouter ------------------------------------------------------------

def test_router_dispatches_accept_and_review():
    accepted, review = [], []
    router = ReviewRouter(
        on_accept=lambda r: accepted.append(r["event_id"]),
        on_review=lambda r, reasons: review.append(reasons),
    )
    router.route(_good_record())
    router.route(dict(_good_record(), tx_type="CRYPTO"))

    assert len(accepted) == 1
    assert len(review) == 1
    assert review[0] == ["unknown_enum:tx_type=CRYPTO"]


def test_router_returns_result():
    router = ReviewRouter(on_accept=lambda r: None, on_review=lambda r, x: None)
    assert router.route(_good_record()).accepted
