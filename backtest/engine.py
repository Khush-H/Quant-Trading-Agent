"""Event-driven spot backtester.

Replays historical bars one at a time, in order, and NEVER reads a bar it has
not "reached" yet. A signal attached to the bar that closes at time ``t`` is the
model's view given information available at ``t``; it is therefore executed at
the NEXT bar's open price (``t+1`` open), never at the close that produced it —
trading at the signal bar's own close would be look-ahead, since that close is
only known after the bar completes.

Position state is spot-only and mirrors :class:`core.position.Position`
exactly: ``quantity >= 0``, flat is ``quantity == 0``, there is no short side.
The transitions are:

* LONG signal while flat  -> buy with ~20% of equity (fixed-fractional),
  rounded to the exchange step size, rejected if below minNotional.
* FLAT signal while long  -> sell the entire holding.
* otherwise               -> hold (no pyramiding, no adding to a position).

Costs (taker fee per side + slippage, and the minNotional gate) come solely
from :mod:`backtest.costs`, so the simulation pays exactly what the eventual
paper/live path will. This backtester does NOT depend on the risk engine or
``core.engine.submit_order`` — those are later phases and are not needed to
measure edge.

TODO(exec phase): when the SimulatedExecutor lands, route fills through
``core.engine.submit_order`` so the backtest shares the live risk path (see the
gateway contract). Until then the engine applies fills directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.metrics import buy_and_hold_metrics, compute_metrics
from config import Settings, get_settings
from ml.labels import FLAT, LONG

# Fraction of current equity to deploy on a new long (fixed-fractional sizing).
DEFAULT_POSITION_FRACTION = 0.20


@dataclass
class _SpotPosition:
    """Backtest-internal spot holding. Mirrors core.position.Position.

    quantity >= 0; flat is 0; no short. Kept separate from the persistent
    dataclass to avoid importing trading state into the simulator, but the
    semantics are identical so simulation and the future live path agree.
    """

    quantity: float = 0.0
    avg_entry_price: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("Spot position cannot be short (quantity < 0).")

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.0


@dataclass
class BacktestResult:
    """Container for equity curve, trades, and metrics."""

    equity_curve: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None
    metrics: Optional[dict] = None
    benchmark: Optional[dict] = None
    timeframe: str = "1h"


def run_backtest(
    ohlcv: pd.DataFrame,
    signals: pd.Series,
    settings: Optional[Settings] = None,
    *,
    cost_model: Optional[CostModel] = None,
    timeframe: str = "1h",
    initial_equity: float = 10_000.0,
    position_fraction: float = DEFAULT_POSITION_FRACTION,
) -> BacktestResult:
    """Run the spot backtest over ``ohlcv`` driven by ``signals``.

    Args:
        ohlcv: bars indexed by ts (ascending), columns open/high/low/close/volume.
        signals: per-bar target state aligned to ``ohlcv`` index. Each value is
            ``LONG`` (1) = want to be long, or ``FLAT`` (0) = want to be flat.
            The signal at bar t reflects information through t and is acted on
            at the t+1 open. Bars with no signal are treated as hold/flat-intent
            but never trigger a forced exit on their own.
        cost_model: cost + exchange-filter model; defaults to a 10bps/side model
            seeded from ``settings.slippage_bps``.

    Returns a :class:`BacktestResult` with a net-of-cost equity curve, a trade
    log (one row per fill, with realized ``pnl`` on sells), and metrics plus the
    buy-and-hold benchmark over the same window.
    """
    settings = settings or get_settings()
    if cost_model is None:
        cost_model = CostModel(slippage_bps=settings.slippage_bps)

    if not ohlcv.index.is_monotonic_increasing:
        ohlcv = ohlcv.sort_index()
    signals = signals.reindex(ohlcv.index)

    cash = float(initial_equity)
    pos = _SpotPosition()
    trades: list[dict] = []
    equity_index: list = []
    equity_values: list[float] = []

    index = ohlcv.index
    opens = ohlcv["open"].to_numpy(dtype=float)
    closes = ohlcv["close"].to_numpy(dtype=float)
    n = len(index)

    for i in range(n):
        ts = index[i]
        # Mark-to-market equity at THIS bar's close (information known at i).
        equity = cash + pos.quantity * closes[i]
        equity_index.append(ts)
        equity_values.append(equity)

        # Decide using the signal at bar i, but execute at bar i+1's open so we
        # never trade on a price we only learned by seeing bar i complete.
        if i + 1 >= n:
            continue  # last bar: nothing to execute against
        sig = signals.iloc[i]
        if pd.isna(sig):
            continue
        sig = int(sig)
        exec_price = opens[i + 1]
        exec_ts = index[i + 1]

        if sig == LONG and not pos.is_long:
            _try_buy(cash, pos, equity, exec_price, exec_ts, cost_model,
                     position_fraction, trades)
            cash = _cash_after_last(trades, cash, "buy")
        elif sig == FLAT and pos.is_long:
            cash = _sell_all(cash, pos, exec_price, exec_ts, cost_model, trades)

    equity_curve = pd.Series(equity_values, index=pd.Index(equity_index, name="ts"))
    trades_df = pd.DataFrame(
        trades,
        columns=["ts", "side", "quantity", "price", "notional", "cost", "pnl"],
    )
    metrics = compute_metrics(equity_curve, trades_df, timeframe=timeframe)
    benchmark = buy_and_hold_metrics(ohlcv["close"], timeframe=timeframe)
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades_df,
        metrics=metrics,
        benchmark=benchmark,
        timeframe=timeframe,
    )


def _try_buy(cash, pos, equity, price, ts, cost_model, fraction, trades) -> None:
    """Open a long with ~``fraction`` of equity, if it clears the filters."""
    budget = equity * fraction
    # Reserve room for the fee so cash never goes negative.
    fee_factor = 1.0 + cost_model.taker_fee + cost_model.slippage_bps / 10_000.0
    raw_qty = budget / (price * fee_factor)
    qty = cost_model.round_qty(raw_qty)
    if not cost_model.can_trade(qty, price):
        return  # below minNotional / minQty -> no trade (not a resized one)
    notional = qty * price
    cost = cost_model.fill_cost(notional)
    if notional + cost > cash + 1e-9:
        return  # not enough cash after costs
    pos.quantity = qty
    pos.avg_entry_price = price
    trades.append({
        "ts": ts, "side": "buy", "quantity": qty, "price": price,
        "notional": notional, "cost": cost, "pnl": np.nan,
    })


def _cash_after_last(trades, cash, side) -> float:
    """Apply the cash effect of the just-appended buy fill."""
    if not trades or trades[-1]["side"] != side:
        return cash
    t = trades[-1]
    return cash - t["notional"] - t["cost"]


def _sell_all(cash, pos, price, ts, cost_model, trades) -> float:
    """Liquidate the entire holding, realizing PnL net of both legs' costs."""
    qty = pos.quantity
    notional = qty * price
    cost = cost_model.fill_cost(notional)
    proceeds = notional - cost
    # Realized PnL net of costs: the buy leg's cost was already paid in cash; we
    # reconstruct round-trip PnL as proceeds minus the entry value+fee.
    entry_notional = qty * pos.avg_entry_price
    entry_cost = cost_model.fill_cost(entry_notional)
    pnl = proceeds - (entry_notional + entry_cost)
    trades.append({
        "ts": ts, "side": "sell", "quantity": qty, "price": price,
        "notional": notional, "cost": cost, "pnl": pnl,
    })
    pos.quantity = 0.0
    pos.avg_entry_price = 0.0
    return cash + proceeds
