"""The single order gateway.

Every order in the system — simulated (backtest), paper, or live — MUST pass
through :func:`submit_order`. This is the one place that:

  1. validates the order,
  2. runs the risk layer (which can reject or resize), and
  3. dispatches to the mode-appropriate executor.

No other module should call an executor's ``execute`` method directly. Keeping
a single chokepoint is what makes the risk layer impossible to bypass: there is
nowhere else an order can originate. The scaffolding here intentionally raises
``NotImplementedError`` in the executors — trading logic is added later per the
README build order (backtest -> ... -> paper -> risk -> ... -> live).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from config import Mode, Settings, get_settings


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Order:
    """An order request, before it has been sent anywhere.

    Frozen so the same object cannot be mutated after risk approval. If the
    risk layer needs to resize, it returns a *new* Order via ``with_quantity``.
    """

    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    client_id: Optional[str] = None

    def with_quantity(self, quantity: float) -> "Order":
        """Return a copy with a different quantity (used by risk resizing)."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            quantity=quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            client_id=self.client_id,
        )


@dataclass(frozen=True)
class OrderResult:
    """The outcome of attempting to place an order."""

    order: Order
    accepted: bool
    mode: Mode
    filled_quantity: float = 0.0
    avg_price: Optional[float] = None
    exchange_order_id: Optional[str] = None
    reason: Optional[str] = None  # populated when accepted is False
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@runtime_checkable
class OrderExecutor(Protocol):
    """Backend that actually places an (approved) order.

    There is one implementation per mode:
      - backtest -> SimulatedExecutor (fills against historical data)
      - paper    -> PaperExecutor (simulated fills against live prices)
      - live     -> LiveExecutor (real orders via ccxt)

    Implementations are added later; the gateway only depends on this protocol.
    """

    mode: Mode

    def execute(self, order: Order) -> OrderResult:  # pragma: no cover - scaffold
        ...


class RiskRejection(Exception):
    """Raised internally when the risk layer rejects an order outright."""


def submit_order(
    order: Order,
    executor: OrderExecutor,
    *,
    risk=None,
    settings: Optional[Settings] = None,
) -> OrderResult:
    """THE one entry point for placing any order, in any mode.

    Args:
        order: The requested order.
        executor: A mode-appropriate :class:`OrderExecutor`.
        risk: A risk engine exposing ``check(order, ...) -> RiskDecision``.
            If ``None``, the default engine from ``core.risk`` is used. The
            parameter exists for dependency injection in tests, never to skip
            risk — passing an object that approves everything is a code-review
            red flag.
        settings: Optional settings override (defaults to ``get_settings()``).

    Returns:
        An :class:`OrderResult`. If risk rejects the order, ``accepted`` is
        False and no executor call is made.

    Contract:
        Risk runs BEFORE the executor, unconditionally. The executor must never
        be invoked for an order the risk layer did not approve.
    """
    settings = settings or get_settings()

    # Defer import to avoid a circular dependency at module load time.
    if risk is None:
        from core.risk import RiskEngine

        risk = RiskEngine(settings=settings)

    # Sanity: the executor's mode must match the configured mode. This stops a
    # live executor from ever being driven by a backtest config, or vice versa.
    if executor.mode is not settings.mode:
        return OrderResult(
            order=order,
            accepted=False,
            mode=settings.mode,
            reason=(
                f"Executor mode {executor.mode.value!r} does not match "
                f"configured MODE {settings.mode.value!r}."
            ),
        )

    decision = risk.check(order, settings=settings)
    if not decision.approved:
        return OrderResult(
            order=order,
            accepted=False,
            mode=settings.mode,
            reason=decision.reason,
        )

    # Risk may have resized the order down (e.g. to respect notional caps).
    approved_order = decision.order

    return executor.execute(approved_order)


# ---------------------------------------------------------------------------
# Paper-trading: risk_check stub + the TradingDaemon.
# ---------------------------------------------------------------------------
import logging  # noqa: E402  (kept local to the paper section for clarity)
from dataclasses import dataclass as _dataclass

logger = logging.getLogger(__name__)


@_dataclass(frozen=True)
class RiskCheckResult:
    """Outcome of :func:`risk_check`. ``approved`` gates the order path."""

    approved: bool
    order: Order
    reason: Optional[str] = None


def risk_check(
    order: Order,
    *,
    settings: Optional[Settings] = None,
    db=None,
    nav: Optional[float] = None,
    current_exposure: float = 0.0,
    now_ms: Optional[int] = None,
) -> RiskCheckResult:
    """The single risk chokepoint every daemon order must pass through.

    Delegates to :meth:`core.risk.RiskEngine.approve` — the real rules
    (SYSTEM_HALT circuit breaker, per-trade and total-exposure caps) live there.
    The chokepoint INTERFACE is stable: callers still pass an ``order`` and get
    a :class:`RiskCheckResult`; the daemon additionally supplies ``db``/``nav``/
    ``current_exposure`` so sizing caps have a reference. Do not weaken or
    bypass this — risk runs before the executor, unconditionally, for every
    order including the HALT flatten-sell.
    """
    from core.risk import RiskEngine

    engine = RiskEngine(settings=settings, db=db)
    decision = engine.approve(
        order, nav=nav, current_exposure=current_exposure, now_ms=now_ms,
    )
    logger.info(
        "risk_check: %s %s %s qty=%s -> %s (%s)",
        "APPROVE" if decision.approved else "REJECT",
        order.side.value, order.symbol, order.quantity,
        decision.approved, decision.reason,
    )
    return RiskCheckResult(
        approved=decision.approved, order=decision.order, reason=decision.reason,
    )


class TradingDaemon:
    """PAPER-mode decision loop: wake hourly, infer, route deltas to a fill.

    One cycle: fetch OHLCV -> DROP the forming candle -> build features on the
    last CLOSED candle -> load the model (if present) -> infer P(Long). Act only
    when confidence > threshold AND :func:`risk_check` approves. "Act" means
    compute the target position, diff it via :class:`core.position.PositionManager`,
    and route the delta through the paper executor. EVERY decision — including
    holds/no-trades — is written to ``execution_logs`` with the simulated fee
    and slippage.

    PAPER ONLY. The daemon constructs the paper executor via the mode-keyed
    factory; it never instantiates a live executor and there is no live path.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        *,
        settings: Optional[Settings] = None,
        db=None,
        position_manager=None,
        executor=None,
        predictor=None,
        model_name: str = "paper_model",
        equity: float = 10_000.0,
    ) -> None:
        from config import get_settings as _get_settings
        from core.database import Database

        self.settings = settings or _get_settings()
        if self.settings.mode is not Mode.PAPER:
            raise ValueError(
                f"TradingDaemon is PAPER-only; configured mode is "
                f"{self.settings.mode.value!r}. Set MODE=paper."
            )
        self.symbol = symbol
        self.timeframe = timeframe
        self.db = db or Database(self.settings)
        self.db.init_schema()

        from core.exchange import executor_for
        from core.position import PositionManager

        self.position_manager = position_manager or PositionManager(self.db)
        self.executor = executor or executor_for(self.settings)
        # `predictor` is an injectable callable(features_row) -> P(Long); when
        # None the daemon loads the committed model lazily (see _infer).
        self._predictor = predictor
        self.model_name = model_name
        self.equity = equity

    # --- one decision cycle -------------------------------------------------
    def run_once(self, ohlcv) -> dict:
        """Run a single decision cycle on the supplied OHLCV frame.

        ``ohlcv`` is the raw fetch (newest bar may be the forming one). Returns a
        summary dict of the decision; also persists it to execution_logs.
        """
        import pandas as pd

        from ml.features import build_features

        if ohlcv is None or len(ohlcv) == 0:
            return self._log_hold(ts=None, confidence=None, price=None,
                                  reason="no OHLCV available")

        # DROP the forming candle: the last row is the in-progress bar.
        closed = ohlcv.iloc[:-1] if len(ohlcv) > 1 else ohlcv.iloc[0:0]
        if len(closed) == 0:
            return self._log_hold(ts=None, confidence=None, price=None,
                                  reason="no closed candle after dropping forming bar")

        feats = build_features(closed)
        if len(feats) == 0:
            return self._log_hold(ts=int(closed.index[-1]), confidence=None,
                                  price=float(closed["close"].iloc[-1]),
                                  reason="insufficient history to build features")

        last_ts = int(feats.index[-1])
        last_close = float(closed.loc[last_ts, "close"])
        feature_row = feats.loc[[last_ts]]

        # Heartbeat + NAV bookkeeping for the circuit breaker. Recorded every
        # cycle BEFORE any decision so the breaker sees current state.
        held_qty = self.position_manager.current_quantity(self.symbol)
        nav = self._nav(held_qty, last_close)
        self.db.record_heartbeat(last_ts)
        self.db.record_nav(last_ts, nav)

        # Evaluate the circuit breaker. If halted, flatten to cash (if holding)
        # and refuse new entries. The flatten SELL still routes through the
        # chokepoint — no bypass.
        from core.risk import RiskEngine
        from core.position import Action

        halt_reason = RiskEngine(settings=self.settings, db=self.db).evaluate_halt(
            now_ms=last_ts
        )
        if halt_reason is not None:
            if held_qty > 0:
                return self._route(
                    Action.SELL, held_qty, ts=last_ts, confidence=None,
                    ref_price=last_close, nav=nav,
                    current_exposure=held_qty * last_close,
                    intent=f"HALT flatten-to-cash ({halt_reason})",
                )
            return self._log_hold(ts=last_ts, confidence=None, price=last_close,
                                  reason=f"SYSTEM_HALT active: {halt_reason}")

        proba = self._infer(feature_row, ts=last_ts, price=last_close)
        if proba is None:  # no model available -> hold already logged
            return self._log_hold(ts=last_ts, confidence=None, price=last_close,
                                  reason="no committed model available")

        # Decide target: long the full sized position iff confident enough.
        threshold = self.settings.confidence_threshold
        want_long = proba > threshold
        target_qty = self._sized_qty(last_close) if want_long else 0.0

        delta = self.position_manager.delta_to(self.symbol, target_qty)

        if delta.action is Action.HOLD:
            return self._log_hold(ts=last_ts, confidence=proba, price=last_close,
                                  reason=f"target matches holding (p={proba:.3f})")

        return self._route(
            delta.action, delta.quantity, ts=last_ts, confidence=proba,
            ref_price=last_close, nav=nav,
            current_exposure=held_qty * last_close,
            intent="signal",
        )

    def _route(self, action, quantity, *, ts, confidence, ref_price, nav,
               current_exposure, intent) -> dict:
        """Route ONE order through the chokepoint, then the executor on approval."""
        from core.position import Action

        side = Side.BUY if action is Action.BUY else Side.SELL
        order = Order(symbol=self.symbol, side=side, quantity=quantity,
                      limit_price=ref_price)
        # The SINGLE risk chokepoint. Nothing reaches the executor without it.
        decision = risk_check(
            order, settings=self.settings, db=self.db, nav=nav,
            current_exposure=current_exposure, now_ms=ts,
        )
        if not decision.approved:
            return self._log(action=action.value, ts=ts, confidence=confidence,
                             price=ref_price, quantity=0.0, notional=0.0,
                             fee=0.0, slippage=0.0, accepted=False,
                             reason=f"risk_check rejected: {decision.reason}")
        result = self.executor.execute(decision.order)
        return self._apply_result(action, result, ts=ts, confidence=confidence,
                                  ref_price=ref_price)

    # --- helpers ------------------------------------------------------------
    def _infer(self, feature_row, *, ts: int, price: float):
        """Return P(Long) for the feature row, or None if no model is available."""
        if self._predictor is not None:
            return float(self._predictor(feature_row))
        # Lazy-load the committed model; hold if there isn't one.
        from core.database import FEATURE_COLUMNS
        from ml.train import FeatureMismatchError, load_model

        try:
            model = load_model(self.model_name, list(FEATURE_COLUMNS))
        except FileNotFoundError:
            return None
        except FeatureMismatchError as exc:
            logger.warning("Model load refused: %s", exc)
            self._log_hold(ts=ts, confidence=None, price=price,
                           reason=f"model feature mismatch: {exc}")
            return None
        self._predictor = lambda row: model.predict_proba(row)[:, 1][0]
        return float(self._predictor(feature_row))

    def _sized_qty(self, price: float) -> float:
        """Fixed-fractional target size (20% of equity), matching the backtest."""
        from backtest.engine import DEFAULT_POSITION_FRACTION

        if price <= 0:
            return 0.0
        return (self.equity * DEFAULT_POSITION_FRACTION) / price

    def _nav(self, held_qty: float, price: float) -> float:
        """Account NAV = starting equity + realized PnL + unrealized PnL.

        Equivalent to cash + position value. Drives the drawdown breaker and the
        percentage sizing caps. Reads realized PnL / avg entry from the stored
        position so it survives restarts.
        """
        pos = self.position_manager.current(self.symbol)
        unrealized = (price - pos.avg_entry_price) * held_qty if held_qty > 0 else 0.0
        return self.equity + pos.realized_pnl + unrealized

    def _apply_result(self, action, result: OrderResult, *, ts, confidence,
                      ref_price) -> dict:
        from core.position import Action

        if not result.accepted:
            return self._log(action=action.value, ts=ts, confidence=confidence,
                             price=ref_price, quantity=0.0, notional=0.0,
                             fee=0.0, slippage=0.0, accepted=False,
                             reason=result.reason)
        qty = result.filled_quantity
        fill_price = result.avg_price or ref_price
        notional = qty * ref_price
        # Recompute the simulated fee/slippage from the shared cost model so the
        # logged numbers reconcile to the backtester exactly.
        cm = getattr(self.executor, "cost_model", None)
        if cm is not None:
            fee = notional * cm.taker_fee
            slippage = notional * (cm.slippage_bps / 10_000.0)
        else:
            fee = slippage = 0.0
        # Persist the holding change.
        self.position_manager.apply_fill(self.symbol, action, qty, fill_price)
        return self._log(action=action.value, ts=ts, confidence=confidence,
                         price=fill_price, quantity=qty, notional=notional,
                         fee=fee, slippage=slippage, accepted=True,
                         reason="paper fill (simulated)")

    def _log_hold(self, *, ts, confidence, price, reason) -> dict:
        return self._log(action="hold", ts=ts, confidence=confidence,
                         price=price, quantity=0.0, notional=0.0, fee=0.0,
                         slippage=0.0, accepted=True, reason=reason)

    def _log(self, *, action, ts, confidence, price, quantity, notional, fee,
             slippage, accepted, reason) -> dict:
        self.db.log_execution(
            mode=self.settings.mode.value, symbol=self.symbol,
            timeframe=self.timeframe, action=action, ts=ts,
            confidence=confidence, price=price, quantity=quantity,
            notional=notional, fee=fee, slippage=slippage, accepted=accepted,
            reason=reason,
        )
        summary = {
            "action": action, "ts": ts, "confidence": confidence,
            "price": price, "quantity": quantity, "notional": notional,
            "fee": fee, "slippage": slippage, "accepted": accepted,
            "reason": reason,
        }
        logger.info("decision: %s", summary)
        return summary
