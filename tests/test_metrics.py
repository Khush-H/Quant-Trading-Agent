"""Tests for performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    buy_and_hold_metrics,
    compute_metrics,
    max_drawdown,
    total_return,
)


def test_total_return():
    eq = pd.Series([100.0, 110.0, 121.0])
    assert total_return(eq) == pytest.approx(0.21)


def test_max_drawdown_is_worst_peak_to_trough():
    eq = pd.Series([100.0, 120.0, 60.0, 90.0])  # peak 120 -> trough 60 = -50%
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_no_drawdown_on_monotonic_curve():
    eq = pd.Series([100.0, 101.0, 102.0])
    assert max_drawdown(eq) == pytest.approx(0.0)


def test_compute_metrics_counts_trades_and_winrate():
    eq = pd.Series([10_000.0, 10_050.0, 10_020.0],
                   index=pd.Index([0, 1, 2], name="ts"))
    trades = pd.DataFrame({
        "ts": [1, 2],
        "side": ["buy", "sell"],
        "quantity": [0.1, 0.1],
        "price": [100.0, 110.0],
        "notional": [10.0, 11.0],
        "cost": [0.01, 0.011],
        "pnl": [np.nan, 0.5],  # one closed (winning) round trip
    })
    m = compute_metrics(eq, trades, timeframe="1h")
    assert m["num_trades"] == 1          # only the closed (sell) leg counts
    assert m["win_rate"] == pytest.approx(1.0)
    assert m["avg_trade_pnl"] == pytest.approx(0.5)
    assert m["turnover"] > 0


def test_buy_and_hold_benchmark():
    close = pd.Series([100.0, 200.0], index=pd.Index([0, 1], name="ts"))
    b = buy_and_hold_metrics(close, timeframe="1h")
    assert b["total_return"] == pytest.approx(1.0)  # doubled
