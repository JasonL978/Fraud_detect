"""Replay PaySim into the transactions.v1 topic, Avro-serialized via the registry.

Run (requires the Day 1 stack up and data/raw/paysim.csv present):

    python data/generators/replay.py --limit 1000 --rate 500

Defaults target the local dev stack from docker-compose.yml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from confluent_kafka import Producer  # noqa: E402
from confluent_kafka.schema_registry import SchemaRegistryClient  # noqa: E402
from confluent_kafka.schema_registry.avro import AvroSerializer  # noqa: E402
from confluent_kafka.serialization import (  # noqa: E402
    MessageField,
    SerializationContext,
    StringSerializer,
)

from fraud_detect.replay_source import RatePacer, iter_mapped_records, message_key  # noqa: E402

_SCHEMA_PATH = _REPO_ROOT / "data" / "schemas" / "transaction_v1.avsc"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Replay PaySim into transactions.v1")
    p.add_argument("--csv", default=str(_REPO_ROOT / "data" / "raw" / "paysim.csv"))
    p.add_argument("--bootstrap", default="localhost:19092")
    p.add_argument("--registry", default="http://localhost:18081")
    p.add_argument("--topic", default="transactions.v1")
    p.add_argument("--rate", type=float, default=1000.0, help="records/sec; 0 = unlimited")
    p.add_argument("--limit", type=int, default=None, help="stop after N records")
    p.add_argument("--poll-every", type=int, default=1000, help="serve callbacks every N records")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    schema_str = _SCHEMA_PATH.read_text(encoding="utf-8")
    sr_client = SchemaRegistryClient({"url": args.registry})
    avro_serializer = AvroSerializer(sr_client, schema_str, lambda rec, ctx: rec)
    key_serializer = StringSerializer("utf_8")

    producer = Producer({"bootstrap.servers": args.bootstrap})

    stats = {"ok": 0, "err": 0}

    def on_delivery(err, msg):
        if err is not None:
            stats["err"] += 1
            if stats["err"] <= 10:
                print(f"delivery failed: {err}", file=sys.stderr)
        else:
            stats["ok"] += 1

    pacer = RatePacer(args.rate)
    value_ctx = SerializationContext(args.topic, MessageField.VALUE)
    key_ctx = SerializationContext(args.topic, MessageField.KEY)

    produced = 0
    for record in iter_mapped_records(args.csv, limit=args.limit):
        pacer.wait()
        value = avro_serializer(record, value_ctx)
        key = key_serializer(message_key(record), key_ctx)
        stalls = 0
        while True:
            try:
                producer.produce(args.topic, key=key, value=value, on_delivery=on_delivery)
                break
            except BufferError:
                # Local queue full: serve callbacks to drain it, then retry.
                # Bail if it never drains (broker likely unreachable) so we fail
                # loudly instead of hanging forever.
                producer.poll(0.5)
                stalls += 1
                if stalls >= 60:  # ~30s with no progress
                    producer.flush()
                    raise RuntimeError(
                        "producer queue full for ~30s with no delivery; "
                        f"is the broker reachable at {args.bootstrap}?"
                    )
        produced += 1
        if produced % args.poll_every == 0:
            producer.poll(0)

    producer.flush()
    print(f"produced={produced} delivered={stats['ok']} failed={stats['err']}")
    return 1 if stats["err"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
