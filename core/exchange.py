"""Exchange connectivity (ccxt) and the per-mode order executors.

This module holds the three :class:`~core.engine.OrderExecutor` backends. They
are the ONLY things that talk to an exchange or simulate a fill, and they are
only ever invoked by :func:`core.engine.submit_order` after risk approval —
never call ``execute`` directly from strategy code.

Credentials are pulled from :mod:`config.settings`, which sources them
exclusively from environment variables. Nothing here reads ``os.environ``.
"""

from __future__ import annotations

from typing import Optional

from config import Mode, Settings, get_settings
from core.engine import Order, OrderResult


def build_exchange(settings: Optional[Settings] = None):
    """Construct a configured ccxt exchange client.

    Wires sandbox mode and env-sourced credentials. Returns the ccxt instance;
    the import is deferred so that backtests (which need no exchange) don't pay
    for it. Implemented during the data/live phases.
    """
    settings = settings or get_settings()
    # import ccxt  # deferred until implemented
    # creds = settings.exchange_credentials()
    # exchange = getattr(ccxt, settings.exchange_id)({...})
    # if settings.exchange_sandbox: exchange.set_sandbox_mode(True)
    raise NotImplementedError("Exchange client built during data/live phase.")


class SimulatedExecutor:
    """Backtest executor: fills against historical data. mode == BACKTEST."""

    mode = Mode.BACKTEST

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def execute(self, order: Order) -> OrderResult:
        raise NotImplementedError("Simulated fills implemented in backtest phase.")


class PaperExecutor:
    """Paper executor: simulated fills against LIVE prices. mode == PAPER."""

    mode = Mode.PAPER

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def execute(self, order: Order) -> OrderResult:
        raise NotImplementedError("Paper fills implemented in paper phase.")


class LiveExecutor:
    """Live executor: places REAL orders via ccxt. mode == LIVE.

    Constructing this does not bypass the live gate — settings construction
    already refused to start unless LIVE_TRADING_CONFIRMED=true. This is the
    last hop, and it is still only reachable through ``submit_order``.
    """

    mode = Mode.LIVE

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        # self.exchange = build_exchange(self.settings)

    def execute(self, order: Order) -> OrderResult:
        raise NotImplementedError("Live order placement implemented in live phase.")


def executor_for(settings: Optional[Settings] = None):
    """Return the correct executor for the configured MODE.

    This is the recommended way to obtain an executor so that mode and executor
    can never drift apart (``submit_order`` also defends against a mismatch).
    """
    settings = settings or get_settings()
    if settings.mode is Mode.BACKTEST:
        return SimulatedExecutor(settings)
    if settings.mode is Mode.PAPER:
        return PaperExecutor(settings)
    if settings.mode is Mode.LIVE:
        return LiveExecutor(settings)
    raise ValueError(f"Unknown mode: {settings.mode!r}")
