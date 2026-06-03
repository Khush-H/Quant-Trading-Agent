"""Risk layer — the circuit breaker. Reached only via the order chokepoint.

The :class:`RiskEngine` is the single component allowed to approve, reject, or
resize an order. ``core.engine.risk_check`` (the chokepoint the daemon calls)
delegates to :meth:`RiskEngine.approve`, so every order — buy or sell — passes
through here before any executor runs. There is no other path to a fill.

Circuit breaker (SYSTEM_HALT), persisted in ``system_state``:

* Trips if the rolling 24h NAV drawdown is at or below ``halt_drawdown_pct``
  (default −3.0%), OR
* after ``halt_max_consecutive_failures`` consecutive ccxt/exchange failures, OR
* if the heartbeat hasn't updated within ``halt_heartbeat_timeout_minutes``.

Once HALT is set it NEVER self-clears: the engine flattens to cash (paper) and
refuses ALL new entries (BUYs) until an operator runs ``scripts.reset_halt``.
While halted, SELL orders are still approved so the flatten/de-risk routes
through this same chokepoint (no bypass).

Hard sizing limits (always enforced, halted or not):

* per-trade notional ≤ ``max_trade_fraction`` × NAV (default 20%),
* total long exposure ≤ ``max_total_exposure`` × NAV (default 20%).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from config import Settings, get_settings


@dataclass(frozen=True)
class RiskDecision:
    """Result of a risk check.

    ``order`` is the (possibly resized) order to send onward. When ``approved``
    is False, ``reason`` explains why and ``order`` is unchanged.
    """

    approved: bool
    order: "object"  # core.engine.Order — typed as object to avoid a cycle
    reason: Optional[str] = None


def rolling_drawdown_pct(nav_history: list, now_ms: int, window_ms: int) -> Optional[float]:
    """Worst peak-to-current drawdown (%) over the trailing ``window_ms``.

    ``nav_history`` is a list of ``[ts_ms, nav]`` samples. Returns the drawdown
    of the LATEST nav from the rolling peak within the window, as a percentage
    (e.g. -3.0 for a 3% decline), or None if there is too little history.
    """
    pts = [p for p in nav_history if p[0] >= now_ms - window_ms]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    peak = max(p[1] for p in pts)
    current = pts[-1][1]
    if peak <= 0:
        return None
    return (current / peak - 1.0) * 100.0


class RiskEngine:
    """Pre-trade risk checks + the SYSTEM_HALT circuit breaker.

    Stateless sizing checks (per-trade and total-exposure caps) and the stateful
    HALT logic both live here. HALT state, the heartbeat, the exchange-failure
    counter, and the NAV history are persisted in ``system_state`` via the
    database, so they survive across daemon cycles and process restarts.
    """

    def __init__(self, settings: Optional[Settings] = None, db=None) -> None:
        self.settings = settings or get_settings()
        if db is None:
            from core.database import Database

            db = Database(self.settings)
        self.db = db

    # --- circuit-breaker evaluation ----------------------------------------
    def evaluate_halt(self, *, now_ms: Optional[int] = None) -> Optional[str]:
        """Trip HALT if any breaker condition is met; return the reason or None.

        Idempotent and one-directional: it can SET halt but never clears it.
        Safe to call at the top of every cycle.
        """
        if self.db.is_halted():
            return self.db.halt_reason() or "halted"

        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        reason = self._breaker_reason(now_ms)
        if reason is not None:
            self.db.set_halt(reason)
        return reason

    def _breaker_reason(self, now_ms: int) -> Optional[str]:
        s = self.settings
        # 1) Rolling 24h drawdown.
        dd = rolling_drawdown_pct(self.db.nav_history(), now_ms, 24 * 3_600_000)
        if dd is not None and dd <= s.halt_drawdown_pct:
            return (f"rolling 24h drawdown {dd:.2f}% <= "
                    f"{s.halt_drawdown_pct:.2f}% threshold")
        # 2) Consecutive exchange failures.
        fails = self.db.exchange_consecutive_failures()
        if fails >= s.halt_max_consecutive_failures:
            return (f"{fails} consecutive exchange failures >= "
                    f"{s.halt_max_consecutive_failures}")
        # 3) Stale heartbeat.
        hb = self.db.last_heartbeat_ms()
        if hb is not None:
            stale_ms = now_ms - hb
            limit_ms = s.halt_heartbeat_timeout_minutes * 60_000
            if stale_ms > limit_ms:
                return (f"heartbeat stale by {stale_ms/60_000:.1f} min > "
                        f"{s.halt_heartbeat_timeout_minutes:.1f} min")
        return None

    @property
    def halted(self) -> bool:
        return self.db.is_halted()

    # --- the approval chokepoint -------------------------------------------
    def approve(
        self,
        order,
        *,
        nav: Optional[float] = None,
        current_exposure: float = 0.0,
        now_ms: Optional[int] = None,
    ) -> RiskDecision:
        """Approve / reject an order against HALT state and sizing caps.

        ``nav`` is the account net asset value used for the percentage caps;
        ``current_exposure`` is the notional already held. Both buys and sells
        come through here. While halted, BUYs (new entries / adds) are rejected
        and SELLs (flatten / de-risk) are still approved.
        """
        from core.engine import Side  # local import to avoid a cycle

        # Refresh HALT first so a freshly-tripped breaker takes effect now.
        self.evaluate_halt(now_ms=now_ms)

        if order.quantity <= 0:
            return RiskDecision(False, order, "Order quantity must be positive.")

        is_buy = order.side is Side.BUY

        if self.halted:
            if is_buy:
                return RiskDecision(
                    False, order,
                    f"SYSTEM_HALT active ({self.db.halt_reason()}): "
                    "new entries refused until manual reset.",
                )
            # SELL while halted: allow it so we can flatten through the chokepoint.
            return RiskDecision(True, order, "halted: sell (flatten/de-risk) allowed")

        # Sells are always allowed when not halted (reducing risk).
        if not is_buy:
            return RiskDecision(True, order, "sell approved")

        # --- BUY sizing caps (need a NAV reference) ---
        price = order.limit_price
        if nav is None or nav <= 0 or price is None or price <= 0:
            # Without a price/NAV we can't size-check a buy; refuse rather than
            # let an unbounded order through.
            return RiskDecision(
                False, order,
                "cannot size-check buy without a positive NAV and price.",
            )
        notional = order.quantity * price
        max_trade = self.settings.max_trade_fraction * nav
        if notional > max_trade + 1e-9:
            return RiskDecision(
                False, order,
                f"per-trade cap: notional {notional:.2f} > "
                f"{self.settings.max_trade_fraction:.0%} NAV ({max_trade:.2f}).",
            )
        max_exposure = self.settings.max_total_exposure * nav
        if current_exposure + notional > max_exposure + 1e-9:
            return RiskDecision(
                False, order,
                f"total-exposure cap: {current_exposure + notional:.2f} > "
                f"{self.settings.max_total_exposure:.0%} NAV ({max_exposure:.2f}).",
            )
        return RiskDecision(True, order, "buy approved within caps")

    # --- legacy interface used by core.engine.submit_order -----------------
    def check(self, order, *, settings: Optional[Settings] = None) -> RiskDecision:
        """Back-compat shim. ``submit_order`` calls ``check``; route to approve.

        Without NAV context the structural checks (positive qty, HALT side
        rules) still apply; sizing caps that need NAV are deferred to the
        daemon's ``approve`` call, which supplies NAV.
        """
        return self.approve(order)
