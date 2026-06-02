"""Performance metrics.

Sharpe/Sortino, max drawdown, hit rate, turnover, exposure, etc., computed from
an equity curve and trade log. Implemented in the backtest phase.
"""

from __future__ import annotations

import pandas as pd


def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
    """Return a dict of summary performance metrics.

    Annualize using the data's bar frequency; state the risk-free assumption
    explicitly when adding Sharpe so results are reproducible.
    """
    raise NotImplementedError("Metrics implemented in backtest phase.")
