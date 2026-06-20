"""Known-answer tests for the shared aggregation core (Week 2 parity foundation)."""

from __future__ import annotations

import pytest

from fraud_detect.aggregation import (
    DistinctWindowedCounter,
    PairHistory,
    WindowedAggregator,
    flatten,
)

# Small explicit window set so expected values are easy to reason about.
WINDOWS = (("10s", 10_000), ("1m", 60_000))


# --- WindowedAggregator ------------------------------------------------------

def test_windowed_counts_and_stats_known_answer():
    agg = WindowedAggregator(WINDOWS)
    # ts in ms: 0, 5s, 20s ; amounts 100, 200, 50
    agg.add(0, 100.0)
    agg.add(5_000, 200.0)
    agg.add(20_000, 50.0)

    s = agg.query(21_000)
    # 10s window [11000, 21000): only ts=20000 -> amount 50
    assert s["10s"] == (1, 50.0, 50.0, 50.0)
    # 1m window [-39000, 21000): all three -> 100,200,50
    assert s["1m"].count == 3
    assert s["1m"].sum == 350.0
    assert s["1m"].mean == pytest.approx(350.0 / 3)
    assert s["1m"].max == 200.0


def test_empty_window_yields_zero_mean_and_max():
    agg = WindowedAggregator(WINDOWS)
    agg.add(0, 100.0)
    s = agg.query(100_000)  # far past the 1m window -> empty
    assert s["10s"] == (0, 0.0, 0.0, 0.0)
    assert s["1m"] == (0, 0.0, 0.0, 0.0)


def test_current_event_excluded_query_before_add():
    agg = WindowedAggregator(WINDOWS)
    agg.add(0, 100.0)
    # Query as-of t=5000 BEFORE adding the event at 5000.
    s_before = agg.query(5_000)
    assert s_before["10s"].count == 1  # only the prior event at t=0
    agg.add(5_000, 200.0)
    s_after = agg.query(5_001)
    assert s_after["10s"].count == 2


def test_same_ms_prior_event_is_counted():
    # A prior event at exactly the as_of time IS counted (burst capture)...
    agg = WindowedAggregator(WINDOWS)
    agg.add(10_000, 100.0)
    s = agg.query(10_000)
    assert s["10s"].count == 1


def test_current_event_excluded_even_at_same_ms():
    # ...but the CURRENT event is excluded because we query BEFORE adding it,
    # even when a same-ms prior event exists.
    agg = WindowedAggregator(WINDOWS)
    agg.add(10_000, 100.0)        # prior burst member
    s_before = agg.query(10_000)  # scoring a second same-ms event, before add
    assert s_before["10s"].count == 1   # sees the 1 prior, not itself
    agg.add(10_000, 200.0)        # now add the current event
    assert agg.query(10_000)["10s"].count == 2  # a third same-ms event sees both


def test_lower_bound_inclusive():
    agg = WindowedAggregator(WINDOWS)
    agg.add(11_000, 100.0)
    # window 10s for as_of 21000 is [11000, 21000): 11000 is included.
    s = agg.query(21_000)
    assert s["10s"].count == 1


def test_pruning_does_not_drop_needed_events():
    agg = WindowedAggregator(WINDOWS)
    agg.add(0, 10.0)
    agg.add(70_000, 20.0)   # >1m after first; first is now prunable
    s = agg.query(70_001)
    # 1m window [10001, 70001): only the 70000 event
    assert s["1m"].count == 1
    assert s["1m"].sum == 20.0


# --- DistinctWindowedCounter -------------------------------------------------

def test_distinct_fan_in_known_answer():
    fan = DistinctWindowedCounter(WINDOWS)
    fan.add(0, "A")
    fan.add(1_000, "B")
    fan.add(2_000, "A")  # duplicate sender
    out = fan.query(3_000)
    assert out["10s"] == 2          # {A, B}
    assert out["1m"] == 2


def test_distinct_respects_window():
    fan = DistinctWindowedCounter(WINDOWS)
    fan.add(0, "A")
    fan.add(15_000, "B")
    out = fan.query(16_000)
    assert out["10s"] == 1          # only B in last 10s
    assert out["1m"] == 2           # A and B in last 1m


# --- PairHistory -------------------------------------------------------------

def test_pair_history_first_time_then_repeat():
    pair = PairHistory()
    assert pair.query("R1") == (True, 0)
    pair.observe("R1")
    assert pair.query("R1") == (False, 1)
    pair.observe("R1")
    assert pair.query("R1") == (False, 2)
    # A different recipient is still first-time.
    assert pair.query("R2") == (True, 0)


# --- flatten -----------------------------------------------------------------

def test_flatten_naming():
    agg = WindowedAggregator(WINDOWS)
    agg.add(0, 100.0)
    flat = flatten(agg.query(1_000), prefix="sender_")
    assert flat["sender_count_10s"] == 1.0
    assert flat["sender_sum_10s"] == 100.0
    assert "sender_mean_1m" in flat and "sender_max_1m" in flat
    assert all(isinstance(v, float) for v in flat.values())
