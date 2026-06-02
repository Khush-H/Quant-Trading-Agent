"""Tests for the OHLCV ingestion's 'drop incomplete trailing bar' rule.

The exchange returns the currently-forming bar as the last element. Storing it
would let the most recent candle mutate under the backtester, so we drop it on
every fetch until its interval has closed. These tests pin that behavior without
needing a live exchange.
"""

from __future__ import annotations

from scripts.fetch_data import _drop_incomplete_last


def test_drops_last_bar_while_interval_open():
    # 1h bars; last bar opened at 2000ms, closes at 2000+3_600_000. now is inside.
    candles = [[1_000, 1, 1, 1, 1, 1], [3_600_000 + 1_000, 1, 1, 1, 1, 1]]
    now = 3_600_000 + 1_000 + 10  # only 10ms into the last bar
    kept = _drop_incomplete_last(candles, "1h", now)
    assert len(kept) == 1
    assert kept[-1][0] == 1_000


def test_keeps_last_bar_once_interval_closed():
    candles = [[0, 1, 1, 1, 1, 1], [3_600_000, 1, 1, 1, 1, 1]]
    now = 2 * 3_600_000 + 5  # well past the close of the last bar
    kept = _drop_incomplete_last(candles, "1h", now)
    assert len(kept) == 2


def test_empty_input_is_safe():
    assert _drop_incomplete_last([], "1h", 123) == []


def test_unknown_timeframe_drops_last_conservatively():
    candles = [[0, 1, 1, 1, 1, 1], [10, 1, 1, 1, 1, 1]]
    kept = _drop_incomplete_last(candles, "7m", 10_000_000)
    assert len(kept) == 1
