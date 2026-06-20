"""Pure (Kafka-free) building blocks for the PaySim replay producer.

Kept separate from the Kafka wiring so it can be unit-tested without a running
broker. The Kafka CLI lives in ``data/generators/replay.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import pandas as pd

from .paysim_mapping import map_paysim_row

# PaySim columns the mapper depends on.
_REQUIRED_COLUMNS = {
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
}


def iter_mapped_records(
    csv_path: str | Path,
    limit: Optional[int] = None,
    chunksize: int = 50_000,
) -> Iterator[dict]:
    """Yield transactions.v1 records mapped from a PaySim CSV.

    Reads the CSV in chunks (the real file is ~6.3M rows) and assigns each row
    a GLOBAL, monotonically increasing ``row_index`` so deterministic event_ids
    and jitter remain stable across chunk boundaries (parity requirement).
    """
    csv_path = Path(csv_path)
    row_index = 0
    yielded = 0
    first_chunk = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if first_chunk:
            missing = _REQUIRED_COLUMNS - set(chunk.columns)
            if missing:
                raise ValueError(f"PaySim CSV missing columns: {sorted(missing)}")
            first_chunk = False
        for row in chunk.to_dict("records"):
            if limit is not None and yielded >= limit:
                return
            yield map_paysim_row(row, row_index)
            row_index += 1
            yielded += 1


def message_key(record: dict) -> str:
    """Partition key: group all of a sender's transactions on one partition.

    Required for later stateful, per-sender stream features (velocity, etc.).
    """
    return record["sender_id"]


class RatePacer:
    """Pace an iteration to ``rate_per_sec`` records/second.

    ``rate_per_sec <= 0`` disables pacing (run as fast as possible). The clock
    and sleep are injectable for deterministic tests. Tracks an absolute target
    schedule so it does not drift, but never tries to "catch up" by sleeping a
    negative amount.
    """

    def __init__(
        self,
        rate_per_sec: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate_per_sec = rate_per_sec
        self._clock = clock
        self._sleep = sleep
        self._interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._next_at: Optional[float] = None

    def wait(self) -> None:
        """Call once per record to maintain the target rate."""
        if self._interval == 0.0:
            return
        now = self._clock()
        if self._next_at is None:
            self._next_at = now + self._interval
            return
        delay = self._next_at - now
        if delay > 0:
            self._sleep(delay)
            self._next_at += self._interval
        else:
            # We're behind schedule; don't sleep, re-anchor to now to avoid
            # accumulating a burst debt.
            self._next_at = now + self._interval
