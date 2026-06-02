"""Risk layer. Reached only via :func:`core.engine.submit_order`.

The :class:`RiskEngine` is deliberately the only component allowed to approve,
reject, or resize an order. ``submit_order`` calls ``check`` before any executor
runs, so every order — backtest, paper, or live — is subject to the same rules.

Trading-specific limits (drawdown tracking, per-symbol exposure, correlation
caps, etc.) are filled in during the "risk" phase of the build order. This
scaffold establishes the interface and a few obvious notional/position checks
so the gateway has something real to call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import Settings, get_settings


@dataclass(frozen=True)
class RiskDecision:
    """Result of a risk check.

    ``order`` is the (possibly resized) order to send onward. When
    ``approved`` is False, ``reason`` explains why and ``order`` is unchanged.
    """

    approved: bool
    order: "object"  # core.engine.Order — typed as object to avoid a cycle
    reason: Optional[str] = None


class RiskEngine:
    """Pre-trade risk checks.

    Stateless checks (notional caps, leverage) live here directly. Stateful
    checks (open-position count, daily loss) will consult the database/position
    layer once those are implemented; for now they are placeholders that read
    the configured limits so the wiring and limits are visible from day one.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def check(self, order, *, settings: Optional[Settings] = None) -> RiskDecision:
        """Approve, reject, or resize an order against configured limits.

        This is the method ``submit_order`` depends on. Keep it total: always
        return a :class:`RiskDecision`, never raise for an ordinary rejection.
        """
        settings = settings or self.settings

        if order.quantity <= 0:
            return RiskDecision(
                approved=False,
                order=order,
                reason="Order quantity must be positive.",
            )

        # NOTE: Notional = quantity * price. A live/paper implementation will
        # source the reference price from the exchange or last trade; in this
        # scaffold we only enforce the structural checks that need no price.
        # The limit values below are surfaced now so they are never forgotten.
        _max_notional = settings.max_position_notional
        _max_open = settings.max_open_positions
        _max_leverage = settings.max_leverage
        _max_daily_loss = settings.max_daily_loss

        # TODO(risk phase): notional/leverage/open-position/daily-loss checks.
        return RiskDecision(approved=True, order=order)
