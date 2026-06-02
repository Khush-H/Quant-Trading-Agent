"""Tests for the event-driven spot backtester.

Pins the spot-only semantics (quantity never goes negative, no pyramiding),
next-bar-open execution (no look-ahead), and that costs actually reduce equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel, ExchangeFilters
from backtest.engine import run_backtest
from config.settings import Settings
from ml.labels import FLAT, LONG


def _bars(closes, opens=None):
    n = len(closes)
    ts = pd.Index(np.arange(n, dtype=np.int64) * 3_600_000, name="ts")
    opens = closes if opens is None else opens
    return pd.DataFrame(
        {"open": opens, "high": [max(o, c) for o, c in zip(opens, closes)],
         "low": [min(o, c) for o, c in zip(opens, closes)],
         "close": closes, "volume": [100.0] * n},
        index=ts,
    )


def _settings():
    return Settings(_env_file=None, slippage_bps=0.0)


def _cheap_cost():
    # Tiny filters so small synthetic trades are allowed; zero slippage.
    return CostModel(taker_fee=0.0, slippage_bps=0.0,
                     filters=ExchangeFilters(min_notional=1.0, step_size=1e-8, min_qty=0))


def test_position_never_goes_short_and_no_pyramiding():
    # Always-long signal: should buy once, then hold (never add).
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    bars = _bars(closes)
    sig = pd.Series([LONG] * len(closes), index=bars.index)
    res = run_backtest(bars, sig, settings=_settings(), cost_model=_cheap_cost(),
                       timeframe="1h", initial_equity=10_000.0)
    buys = res.trades[res.trades["side"] == "buy"]
    assert len(buys) == 1  # exactly one entry, no pyramiding
    assert (res.trades["quantity"] >= 0).all()  # never short


def test_long_then_flat_round_trips_and_costs_reduce_equity():
    # Buy at bar1 open, sell at bar3 open. Flat price => only costs are lost.
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]
    bars = _bars(closes)
    sig = pd.Series([LONG, LONG, FLAT, FLAT, FLAT], index=bars.index)
    costed = CostModel(taker_fee=0.0010, slippage_bps=0.0,
                       filters=ExchangeFilters(min_notional=1.0, step_size=1e-8, min_qty=0))
    res = run_backtest(bars, sig, settings=_settings(), cost_model=costed,
                       timeframe="1h", initial_equity=10_000.0)
    # One buy, one sell.
    assert set(res.trades["side"]) == {"buy", "sell"}
    # Flat price but we paid two legs of fees -> final equity below start.
    assert res.metrics["final_equity"] < 10_000.0
    # The realized round-trip pnl is negative (pure cost drag).
    sells = res.trades[res.trades["side"] == "sell"]
    assert float(sells["pnl"].iloc[0]) < 0


def test_signal_executes_at_next_open_not_signal_bar_close():
    # Price jumps between bar0 close and bar1 open. A long signal on bar0 must
    # fill at bar1's OPEN (105), proving we don't trade on the bar0 close we
    # only knew after bar0 completed.
    bars = _bars(closes=[100.0, 110.0, 110.0], opens=[100.0, 105.0, 110.0])
    sig = pd.Series([LONG, LONG, LONG], index=bars.index)
    res = run_backtest(bars, sig, settings=_settings(), cost_model=_cheap_cost(),
                       timeframe="1h", initial_equity=10_000.0)
    buy = res.trades[res.trades["side"] == "buy"].iloc[0]
    assert buy["price"] == pytest.approx(105.0)  # next-bar open, not 100 close


def test_last_bar_signal_cannot_trade():
    # A signal only on the final bar has no t+1 to execute against -> no trade.
    bars = _bars(closes=[100.0, 100.0, 100.0])
    sig = pd.Series([FLAT, FLAT, LONG], index=bars.index)
    res = run_backtest(bars, sig, settings=_settings(), cost_model=_cheap_cost(),
                       timeframe="1h")
    assert len(res.trades) == 0


def test_trade_below_min_notional_is_skipped():
    # minNotional very high relative to the 20% budget -> no trade ever opens.
    bars = _bars(closes=[100.0, 100.0, 100.0, 100.0])
    sig = pd.Series([LONG, LONG, LONG, LONG], index=bars.index)
    expensive = CostModel(filters=ExchangeFilters(min_notional=1e9, step_size=1e-8))
    res = run_backtest(bars, sig, settings=_settings(), cost_model=expensive,
                       timeframe="1h", initial_equity=10_000.0)
    assert len(res.trades) == 0
