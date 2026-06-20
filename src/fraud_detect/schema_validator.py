"""Ingest-path schema validator (RT-014).

Why this exists: an unknown categorical value (a new ``tx_type``, or anything
arriving via the HTTP/JSON proxy path that skipped strict Avro decoding) would,
if passed straight to the model, be one-hot-encoded as ALL-ZEROS. The model then
scores it at some arbitrary point — a silent failure and an evasion vector. This
validator checks categoricals against the known vocabulary *before* encoding and
routes unknowns to a review bucket instead.

The known vocabulary is derived from the Avro schema file so it cannot drift from
the contract. The validator is transport-agnostic: it operates on a deserialized
dict, so it guards the Avro path and the JSON/proxy path alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schemas" / "transaction_v1.avsc"

# label_is_fraud is non-nullable in the schema (PaySim is fully labeled), but it
# is the ground-truth ORACLE: an event to be scored legitimately has no label.
# Requiring it on ingest would route every real inference request to review.
_NOT_REQUIRED_ON_INGEST = frozenset({"label_is_fraud"})


@dataclass(frozen=True)
class ValidationResult:
    status: str  # "accept" | "review"
    reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status == "accept"


def _is_nullable(avro_type: Any) -> bool:
    """A field is nullable iff its type is a union containing "null"."""
    return isinstance(avro_type, list) and "null" in avro_type


def _extract_enum_symbols(avro_type: Any) -> list[str] | None:
    """Return enum symbols for a field type (handling union-with-null), else None."""
    if isinstance(avro_type, dict) and avro_type.get("type") == "enum":
        return list(avro_type["symbols"])
    if isinstance(avro_type, list):
        for member in avro_type:
            if isinstance(member, dict) and member.get("type") == "enum":
                return list(member["symbols"])
    return None


@lru_cache(maxsize=1)
def _raw_schema() -> dict:
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def required_fields() -> frozenset[str]:
    """Non-nullable schema fields that must be present on an ingested event,
    excluding the offline-only oracle."""
    fields = _raw_schema()["fields"]
    return frozenset(
        f["name"]
        for f in fields
        if not _is_nullable(f["type"]) and f["name"] not in _NOT_REQUIRED_ON_INGEST
    )


@lru_cache(maxsize=1)
def enum_vocabulary() -> dict[str, frozenset[str]]:
    """Map each enum field name -> its known symbol set, derived from the schema."""
    vocab: dict[str, frozenset[str]] = {}
    for f in _raw_schema()["fields"]:
        symbols = _extract_enum_symbols(f["type"])
        if symbols is not None:
            vocab[f["name"]] = frozenset(symbols)
    return vocab


def validate_event(record: Mapping[str, Any]) -> ValidationResult:
    """Classify an ingested event as accept or review (RT-014).

    Routes to review (never raises, never zero-encodes) when a required field is
    missing/null or an enum field carries a value outside the known vocabulary.
    """
    # A malformed payload may deserialize to None / a non-mapping. The contract
    # is "never raise" -- divert it to review rather than crashing the ingest loop.
    if not isinstance(record, Mapping):
        return ValidationResult("review", ["malformed:not_a_mapping"])

    reasons: list[str] = []

    for name in sorted(required_fields()):
        if record.get(name) is None:
            reasons.append(f"missing_field:{name}")

    for name, symbols in enum_vocabulary().items():
        value = record.get(name)
        if value is None:
            continue  # null is valid for nullable enum fields
        if value not in symbols:
            reasons.append(f"unknown_enum:{name}={value}")

    return ValidationResult("review", reasons) if reasons else ValidationResult("accept")


class ReviewRouter:
    """Route validated events to an accept sink or a review bucket.

    Sinks are callables so this is a plain function now and a Kafka producer
    later (review sink -> transactions.review.v1).
    """

    def __init__(
        self,
        on_accept: Callable[[Mapping[str, Any]], None],
        on_review: Callable[[Mapping[str, Any], list[str]], None],
    ) -> None:
        self._on_accept = on_accept
        self._on_review = on_review

    def route(self, record: Mapping[str, Any]) -> ValidationResult:
        result = validate_event(record)
        if result.accepted:
            self._on_accept(record)
        else:
            self._on_review(record, result.reasons)
        return result
