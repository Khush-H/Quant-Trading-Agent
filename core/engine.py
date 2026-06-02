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
