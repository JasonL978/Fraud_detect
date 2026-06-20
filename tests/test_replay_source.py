"""Tests for the Kafka-free replay building blocks."""

from __future__ import annotations

import csv

import pytest

from fraud_detect.paysim_mapping import make_event_id
from fraud_detect.replay_source import RatePacer, iter_mapped_records, message_key

_HEADER = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def _write_csv(path, n_rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_HEADER)
        for i in range(n_rows):
            w.writerow([
                (i % 744) + 1, "TRANSFER", 100.0 + i, f"C{i}", 100.0 + i, 0.0,
                f"M{i}", 0.0, 0.0, i % 2, 0,
            ])


def test_iter_mapped_records_count_and_limit(tmp_path):
    csv_path = tmp_path / "paysim.csv"
    _write_csv(csv_path, 25)
    assert sum(1 for _ in iter_mapped_records(csv_path)) == 25
    assert sum(1 for _ in iter_mapped_records(csv_path, limit=10)) == 10


def test_global_row_index_is_stable_across_chunks(tmp_path):
    """event_ids must be identical regardless of chunk size (global index)."""
    csv_path = tmp_path / "paysim.csv"
    _write_csv(csv_path, 30)
    big = [r["event_id"] for r in iter_mapped_records(csv_path, chunksize=1000)]
    small = [r["event_id"] for r in iter_mapped_records(csv_path, chunksize=7)]
    assert big == small
    # And they match the deterministic IDs derived from the global index.
    assert big == [make_event_id(i) for i in range(30)]


def test_missing_columns_raise(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "type", "amount"])
        w.writerow([1, "TRANSFER", 10.0])
    with pytest.raises(ValueError, match="missing columns"):
        list(iter_mapped_records(csv_path))


def test_message_key_is_sender(tmp_path):
    csv_path = tmp_path / "paysim.csv"
    _write_csv(csv_path, 3)
    rec = next(iter_mapped_records(csv_path))
    assert message_key(rec) == rec["sender_id"]


# --- RatePacer ---------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, d):
        self.sleeps.append(d)
        self.t += d


def test_rate_pacer_unlimited_never_sleeps():
    clk = FakeClock()
    pacer = RatePacer(0, clock=clk.now, sleep=clk.sleep)
    for _ in range(100):
        pacer.wait()
    assert clk.sleeps == []


def test_rate_pacer_paces_to_interval():
    clk = FakeClock()
    pacer = RatePacer(10, clock=clk.now, sleep=clk.sleep)  # interval 0.1s
    pacer.wait()  # first call primes the schedule, no sleep
    pacer.wait()
    pacer.wait()
    assert clk.sleeps == [pytest.approx(0.1), pytest.approx(0.1)]


def test_rate_pacer_does_not_sleep_when_behind():
    # Clock that jumps far ahead between calls -> we are always behind schedule.
    class JumpClock:
        def __init__(self):
            self.t = 0.0
            self.sleeps = []

        def now(self):
            self.t += 5.0  # 5s elapse between every check
            return self.t

        def sleep(self, d):
            self.sleeps.append(d)

    jc = JumpClock()
    pacer = RatePacer(10, clock=jc.now, sleep=jc.sleep)
    for _ in range(5):
        pacer.wait()
    assert jc.sleeps == []  # never sleeps a negative/zero amount
