"""Shared aggregation primitives for online (Bytewax) and offline (batch).

The same code computes features in both paths, so they cannot drift -- this is
the foundation of the online/offline parity guarantee (Week 2 spine).

Contract for every primitive:
  * Feed events per entity in non-decreasing (timestamp_ms, event_id) order.
  * QUERY BEFORE ADD: compute features from prior events, then add the current
    event. The current event is never part of its own features (no leakage).
  * Windowed membership is ``as_of - w <= ts <= as_of`` over PRIOR events. The
    current event is excluded by query-before-add (it is not yet in the buffer),
    but other events at the same millisecond ARE counted -- otherwise a rapid
    same-instant burst (the velocity-surfing signal) would be undercounted.
  * Empty window -> mean = 0.0, max = 0.0.
"""

from __future__ import annotations

import bisect
from typing import NamedTuple

from . import constants as C


class WindowStat(NamedTuple):
    count: int
    sum: float
    mean: float
    max: float


_DEFAULT_WINDOWS = C.FEATURE_WINDOWS


class WindowedAggregator:
    """Per-entity numeric stats (count/sum/mean/max) over several time windows.

    Used for both sender and recipient velocity. Backed by parallel sorted lists
    and bisect lookups, so add/query are O(log n) amortized.
    """

    def __init__(self, windows=_DEFAULT_WINDOWS) -> None:
        self._windows = tuple(windows)
        self._max_window_ms = max(w for _, w in self._windows)
        self._ts: list[int] = []
        self._amt: list[float] = []

    def add(self, ts_ms: int, amount: float) -> None:
        # Expect non-decreasing ts; bisect_right tolerates minor out-of-order
        # without corrupting sort order.
        i = bisect.bisect_right(self._ts, ts_ms)
        self._ts.insert(i, ts_ms)
        self._amt.insert(i, float(amount))
        self._prune(ts_ms)

    def _prune(self, now_ms: int) -> None:
        # Drop events older than the longest window relative to the latest add;
        # they can never fall inside any future window. Never over-prunes.
        cutoff = now_ms - self._max_window_ms
        k = bisect.bisect_left(self._ts, cutoff)
        if k:
            del self._ts[:k]
            del self._amt[:k]

    def query(self, as_of_ms: int) -> dict[str, WindowStat]:
        out: dict[str, WindowStat] = {}
        hi = bisect.bisect_right(self._ts, as_of_ms)  # ts <= as_of (prior bursts count)
        for label, w in self._windows:
            lo = bisect.bisect_left(self._ts, as_of_ms - w)  # ts >= as_of - w
            amts = self._amt[lo:hi]
            c = len(amts)
            s = float(sum(amts))
            out[label] = WindowStat(
                count=c,
                sum=s,
                mean=(s / c if c else 0.0),
                max=(max(amts) if c else 0.0),
            )
        return out


class DistinctWindowedCounter:
    """Per-entity count of DISTINCT values over time windows (e.g. recipient
    fan-in = distinct senders paying this recipient)."""

    def __init__(self, windows=_DEFAULT_WINDOWS) -> None:
        self._windows = tuple(windows)
        self._max_window_ms = max(w for _, w in self._windows)
        self._ts: list[int] = []
        self._val: list[str] = []

    def add(self, ts_ms: int, value: str) -> None:
        i = bisect.bisect_right(self._ts, ts_ms)
        self._ts.insert(i, ts_ms)
        self._val.insert(i, value)
        cutoff = ts_ms - self._max_window_ms
        k = bisect.bisect_left(self._ts, cutoff)
        if k:
            del self._ts[:k]
            del self._val[:k]

    def query(self, as_of_ms: int) -> dict[str, int]:
        out: dict[str, int] = {}
        hi = bisect.bisect_right(self._ts, as_of_ms)  # ts <= as_of (prior bursts count)
        for label, w in self._windows:
            lo = bisect.bisect_left(self._ts, as_of_ms - w)
            out[label] = len(set(self._val[lo:hi]))
        return out


class PairResult(NamedTuple):
    first_time_payee: bool
    pair_count: int


class PairHistory:
    """One sender's cumulative history of recipients (no time window).

    ``first_time_payee`` is True the first time this sender pays a given
    recipient; ``pair_count`` is how many times they have paid that recipient
    BEFORE the current event (query-before-observe).
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def query(self, recipient_id: str) -> PairResult:
        c = self._counts.get(recipient_id, 0)
        return PairResult(first_time_payee=(c == 0), pair_count=c)

    def observe(self, recipient_id: str) -> None:
        self._counts[recipient_id] = self._counts.get(recipient_id, 0) + 1


def flatten(stats: dict[str, WindowStat], prefix: str) -> dict[str, float]:
    """Flatten windowed stats into a feature vector, e.g. ``sender_count_10s``."""
    out: dict[str, float] = {}
    for label, s in stats.items():
        out[f"{prefix}count_{label}"] = float(s.count)
        out[f"{prefix}sum_{label}"] = s.sum
        out[f"{prefix}mean_{label}"] = s.mean
        out[f"{prefix}max_{label}"] = s.max
    return out
