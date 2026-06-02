"""Performance metrics computed from an equity curve and trade log.

All return-based metrics are derived from the equity curve, which is already
NET OF COSTS (the engine deducts fees+slippage before recording equity). Trade
metrics (win rate, average PnL) likewise use realized PnL net of both legs'
costs. Sharpe/Sortino are annualized from the bar frequency with a zero
risk-free assumption, stated here so results are reproducible.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

# Bars per year for annualization, keyed by timeframe. 24/7 crypto market.
_BARS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}


def _periods_per_year(timeframe: str) -> float:
    return float(_BARS_PER_YEAR.get(timeframe, 8_760))  # default to 1h


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def sharpe(returns: pd.Series, timeframe: str, rf: float = 0.0) -> float:
    """Annualized Sharpe of per-bar returns (zero risk-free by default)."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / _periods_per_year(timeframe)
    sd = excess.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * math.sqrt(_periods_per_year(timeframe)))


def sortino(returns: pd.Series, timeframe: str, rf: float = 0.0) -> float:
    """Annualized Sortino: like Sharpe but penalizes only downside deviation."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / _periods_per_year(timeframe)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    dd = math.sqrt((downside**2).mean())
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * math.sqrt(_periods_per_year(timeframe)))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve, as a fraction <= 0."""
    if len(equity) == 0:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    timeframe: str = "1h",
    rf: float = 0.0,
) -> dict:
    """Return a dict of summary performance metrics.

    ``equity_curve`` is the net-of-cost mark-to-market equity per bar.
    ``trades`` has at least columns ``pnl`` (realized, net of costs) and
    ``notional`` (traded value per fill) for closed round trips / fills.
    """
    eq = equity_curve.dropna()
    bar_returns = eq.pct_change().dropna()

    closed = trades[trades.get("pnl").notna()] if "pnl" in trades else trades
    n_trades = int(len(closed))
    wins = int((closed["pnl"] > 0).sum()) if n_trades else 0
    win_rate = (wins / n_trades) if n_trades else 0.0
    avg_trade_pnl = float(closed["pnl"].mean()) if n_trades else 0.0

    # Turnover: total traded notional / average equity. Counts both legs.
    traded_notional = float(trades["notional"].sum()) if "notional" in trades and len(trades) else 0.0
    avg_equity = float(eq.mean()) if len(eq) else 0.0
    turnover = (traded_notional / avg_equity) if avg_equity else 0.0

    return {
        "total_return": total_return(eq),
        "sharpe": sharpe(bar_returns, timeframe, rf),
        "sortino": sortino(bar_returns, timeframe, rf),
        "max_drawdown": max_drawdown(eq),
        "win_rate": win_rate,
        "avg_trade_pnl": avg_trade_pnl,
        "num_trades": n_trades,
        "turnover": turnover,
        "final_equity": float(eq.iloc[-1]) if len(eq) else 0.0,
    }


def buy_and_hold_metrics(
    close: pd.Series, timeframe: str = "1h", rf: float = 0.0
) -> dict:
    """Benchmark: hold the asset for the whole window (no costs after entry).

    Used by the verdict to compare the strategy against passively holding BTC.
    """
    eq = (close / close.iloc[0]).rename("equity")
    bar_returns = eq.pct_change().dropna()
    return {
        "total_return": total_return(eq),
        "sharpe": sharpe(bar_returns, timeframe, rf),
        "sortino": sortino(bar_returns, timeframe, rf),
        "max_drawdown": max_drawdown(eq),
    }
