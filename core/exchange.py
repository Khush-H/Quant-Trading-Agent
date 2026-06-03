"""Exchange connectivity (ccxt) and the per-mode order executors.

This module holds the :class:`~core.engine.OrderExecutor` backends. They are the
ONLY things that talk to an exchange or simulate a fill, and they are only ever
invoked by :func:`core.engine.submit_order` after risk approval — never call
``execute`` directly from strategy code.

Paper fills are SIMULATED using the backtester's cost model
(:mod:`backtest.costs`): identical taker fee, slippage, and minNotional/stepSize
rounding, so paper and backtest agree by construction. The reference fill price
is supplied by the caller on the order (``limit_price``) — for the daemon that
is the last CLOSED candle's close.

Live execution is present only as a HARD-GATED branch: it refuses to do anything
unless ``LIVE_TRADING_CONFIRMED=true`` (already enforced at settings
construction) AND raises ``NotImplementedError`` regardless, because real order
placement is out of scope for this step. There is no live order logic here.

Credentials are pulled from :mod:`config.settings`, which sources them
exclusively from environment variables. Nothing here reads ``os.environ``.
"""

from __future__ import annotations

from typing import Optional

from backtest.costs import CostModel, ExchangeFilters
from config import Mode, Settings, get_settings
from core.engine import Order, OrderResult, Side


def build_exchange(settings: Optional[Settings] = None):
    """Construct a configured ccxt exchange client (read-only use here).

    Wires sandbox mode and env-sourced credentials. ccxt is imported lazily so
    backtests don't pay for it. Used in paper mode only to load market limits.
    """
    settings = settings or get_settings()
    import ccxt  # deferred: heavy import

    creds = settings.exchange_credentials()
    klass = getattr(ccxt, settings.exchange_id)
    exchange = klass({
        "apiKey": creds["apiKey"],
        "secret": creds["secret"],
        "password": creds["password"],
        "enableRateLimit": True,
    })
    if settings.exchange_sandbox and exchange.has.get("sandbox", False):
        exchange.set_sandbox_mode(True)
    return exchange


def _cost_model(settings: Settings, filters: Optional[ExchangeFilters] = None) -> CostModel:
    """Build the cost model paper shares with the backtester.

    Same per-side taker fee and slippage; filters default to the cost module's
    defaults unless real market limits are supplied.
    """
    return CostModel(
        taker_fee=0.0010,
        slippage_bps=settings.slippage_bps,
        filters=filters or ExchangeFilters(),
    )


class SimulatedExecutor:
    """Backtest executor. mode == BACKTEST.

    The backtest loop applies fills directly via :mod:`backtest.engine`; this
    executor exists for the gateway's mode/executor symmetry and is not the
    backtest fill path in this build.
    """

    mode = Mode.BACKTEST

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def execute(self, order: Order) -> OrderResult:
        raise NotImplementedError(
            "Backtest fills are applied directly in backtest.engine."
        )


class PaperExecutor:
    """Paper executor: simulated fills against a supplied price. mode == PAPER.

    Uses the backtester's :class:`CostModel` so paper fees/slippage and the
    minNotional/stepSize gate match the backtest exactly. The reference fill
    price comes from ``order.limit_price`` (the daemon sets it to the last
    closed close). Quantity is floored to step size; a sub-minNotional order is
    REJECTED, never silently resized.
    """

    mode = Mode.PAPER

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        filters: Optional[ExchangeFilters] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cost_model = _cost_model(self.settings, filters)

    def execute(self, order: Order) -> OrderResult:
        price = order.limit_price
        if price is None or price <= 0:
            return OrderResult(
                order=order, accepted=False, mode=self.mode,
                reason="Paper fill needs a positive reference price (limit_price).",
            )
        qty = self.cost_model.round_qty(order.quantity)
        if not self.cost_model.can_trade(qty, price):
            return OrderResult(
                order=order, accepted=False, mode=self.mode,
                reason=(
                    f"Order below exchange filters: qty={qty} price={price} "
                    f"notional={qty * price:.2f} < minNotional "
                    f"{self.cost_model.filters.min_notional}."
                ),
            )
        notional = qty * price
        # Total simulated cost on this single leg (fee + slippage).
        cost = self.cost_model.fill_cost(notional)
        # Split the reported cost into fee vs slippage for the audit trail.
        fee = notional * self.cost_model.taker_fee
        slippage = notional * (self.cost_model.slippage_bps / 10_000.0)
        # Slippage moves the effective fill price against us.
        slip_per_unit = (slippage / qty) if qty else 0.0
        avg_price = price + slip_per_unit if order.side is Side.BUY else price - slip_per_unit
        return OrderResult(
            order=order.with_quantity(qty),
            accepted=True,
            mode=self.mode,
            filled_quantity=qty,
            avg_price=avg_price,
            exchange_order_id=f"paper-sim-{notional:.2f}",
            reason=None,
        )


class LiveExecutor:
    """Live executor: HARD-GATED, no order logic. mode == LIVE.

    Out of scope for this step. Even though settings construction already refuses
    to start without LIVE_TRADING_CONFIRMED=true, this re-checks the gate and
    then raises unconditionally — there is intentionally no real order placement
    here. The branch exists so the wiring is visible, not so it can run.
    """

    mode = Mode.LIVE

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def execute(self, order: Order) -> OrderResult:
        if not self.settings.live_trading_confirmed:
            return OrderResult(
                order=order, accepted=False, mode=self.mode,
                reason=(
                    "LIVE execution blocked: LIVE_TRADING_CONFIRMED is not true."
                ),
            )
        raise NotImplementedError(
            "Live order placement is out of scope for the paper-trading step. "
            "Implemented later, behind the LIVE_TRADING_CONFIRMED gate."
        )


def executor_for(settings: Optional[Settings] = None):
    """Return the correct executor for the configured MODE.

    Mode and executor can never drift apart this way (``submit_order`` also
    defends against a mismatch).
    """
    settings = settings or get_settings()
    if settings.mode is Mode.BACKTEST:
        return SimulatedExecutor(settings)
    if settings.mode is Mode.PAPER:
        return PaperExecutor(settings)
    if settings.mode is Mode.LIVE:
        return LiveExecutor(settings)
    raise ValueError(f"Unknown mode: {settings.mode!r}")
