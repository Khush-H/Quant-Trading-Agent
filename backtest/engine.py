"""Backtest driver.

Replays historical bars through a strategy and routes every simulated order
through :func:`core.engine.submit_order` with the :class:`SimulatedExecutor`,
so the backtest is subject to the exact same risk layer as paper and live.
That shared path is the whole point: a strategy that survives risk in backtest
behaves identically when promoted. Implemented in the backtest phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import Settings, get_settings


@dataclass
class BacktestResult:
    """Container for equity curve, trades, and metrics. Shape fixed here."""

    equity_curve: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None
    metrics: Optional[dict] = None


def run_backtest(
    ohlcv: pd.DataFrame,
    settings: Optional[Settings] = None,
) -> BacktestResult:
    """Run a backtest over ``ohlcv`` and return results.

    All orders go through ``core.engine.submit_order`` — do not shortcut to the
    executor, or the backtest stops reflecting the live risk rules.
    """
    settings = settings or get_settings()
    raise NotImplementedError("Backtest loop implemented in backtest phase.")
