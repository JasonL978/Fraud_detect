"""Consume transactions.v1, deserialize via the registry, and verify types.

This is the Day 3 "done" check: producer emits, consumer reads N records with
correct types.

Run (after replay.py has produced some records):

    python data/generators/consume_check.py --limit 100
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from confluent_kafka import Consumer  # noqa: E402
from confluent_kafka.schema_registry import SchemaRegistryClient  # noqa: E402
from confluent_kafka.schema_registry.avro import AvroDeserializer  # noqa: E402
from confluent_kafka.serialization import (  # noqa: E402
    MessageField,
    SerializationContext,
)

# Expected (field, python type) for required fields. Tier-2 fields may be None.
_EXPECTED_TYPES = {
    "event_id": str,
    "event_timestamp": dt.datetime,
    "tx_type": str,
    "amount": float,
    "sender_id": str,
    "recipient_id": str,
    "sender_old_balance": float,
    "sender_new_balance": float,
    "recipient_old_balance": float,
    "recipient_new_balance": float,
    "source_step": int,
    "label_is_fraud": bool,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Consume + verify transactions.v1")
    p.add_argument("--bootstrap", default="localhost:19092")
    p.add_argument("--registry", default="http://localhost:18081")
    p.add_argument("--topic", default="transactions.v1")
    p.add_argument("--limit", type=int, default=100, help="records to read")
    p.add_argument("--group", default="consume-check")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for messages")
    return p.parse_args(argv)


def check_types(record: dict) -> list[str]:
    problems = []
    for field, expected in _EXPECTED_TYPES.items():
        if field not in record:
            problems.append(f"missing field {field}")
        elif not isinstance(record[field], expected):
            problems.append(
                f"{field}: expected {expected.__name__}, got {type(record[field]).__name__}"
            )
    return problems


def main(argv=None) -> int:
    args = parse_args(argv)

    sr_client = SchemaRegistryClient({"url": args.registry})
    avro_deserializer = AvroDeserializer(sr_client)

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": args.group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([args.topic])
    value_ctx = SerializationContext(args.topic, MessageField.VALUE)

    read = 0
    bad = 0
    idle = 0.0
    try:
        while read < args.limit:
            msg = consumer.poll(1.0)
            if msg is None:
                idle += 1.0
                if idle >= args.timeout:
                    print(f"no more messages after {args.timeout}s; stopping early")
                    break
                continue
            if msg.error():
                print(f"consume error: {msg.error()}", file=sys.stderr)
                continue
            idle = 0.0
            record = avro_deserializer(msg.value(), value_ctx)
            problems = check_types(record)
            if problems:
                bad += 1
                print(f"BAD record {record.get('event_id')}: {problems}", file=sys.stderr)
            elif read < 3:
                print(f"OK  {record['event_id']} {record['tx_type']} "
                      f"${record['amount']} fraud={record['label_is_fraud']} "
                      f"@ {record['event_timestamp'].isoformat()}")
            read += 1
    finally:
        consumer.close()

    print(f"read={read} type_errors={bad}")
    return 1 if (bad or read == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
