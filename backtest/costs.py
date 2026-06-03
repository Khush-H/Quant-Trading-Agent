"""Trading cost model — the single source of truth for simulated costs.

Backtests without realistic costs are the most common way to fool yourself, so
this is a first-class module rather than a constant buried in the engine. The
eventual paper/live simulator MUST reuse this same :class:`CostModel` so the
numbers a strategy shows in backtest are the numbers it pays when promoted.

Cost of a fill = taker fee on the traded notional + slippage on the traded
notional. Fees are charged PER SIDE: a round trip (buy then sell) pays the
taker fee twice, once on each leg, because each leg is its own fill.

Exchange filters (Binance spot): a fill below ``min_notional`` is rejected (the
exchange would reject it too), and quantities are floored to ``step_size`` and
prices are not sub-``tick_size``. Sizing that rounds a trade below minNotional
means "no trade", never a silently-resized one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExchangeFilters:
    """Binance-style spot symbol filters used to validate a simulated fill.

    Defaults are typical BTC/USDT spot values; override per symbol when known.
    """

    min_notional: float = 10.0   # MIN_NOTIONAL: smallest allowed order value
    step_size: float = 1e-5      # LOT_SIZE stepSize: quantity granularity
    min_qty: float = 1e-5        # LOT_SIZE minQty
    tick_size: float = 0.01      # PRICE_FILTER tickSize

    def round_qty(self, qty: float) -> float:
        """Floor a quantity to a whole multiple of ``step_size``.

        Floor (not round) so a simulated fill is never larger than what sizing
        intended — rounding up could push notional past a cap or buy size you
        don't have cash for.
        """
        if self.step_size <= 0:
            return qty
        steps = math.floor(qty / self.step_size + 1e-9)
        return steps * self.step_size

    def meets_min_notional(self, qty: float, price: float) -> bool:
        return qty > 0 and (qty * price) >= self.min_notional and qty >= self.min_qty

    @classmethod
    def from_ccxt_market(cls, market: dict) -> "ExchangeFilters":
        """Build filters from a ccxt ``market`` dict (limits/precision).

        Falls back to the class defaults for any field the exchange omits, so a
        sparse or partial market description still yields usable filters. This
        is the ONE place a live/paper executor sources its limits; the cost math
        and the rounding/minNotional gate remain in this module so paper and
        backtest agree by construction.
        """
        d = cls()
        limits = (market or {}).get("limits", {}) or {}
        precision = (market or {}).get("precision", {}) or {}
        cost_min = (limits.get("cost", {}) or {}).get("min")
        amt_min = (limits.get("amount", {}) or {}).get("min")
        price_min = (limits.get("price", {}) or {}).get("min")
        # ccxt precision.amount may be a step size (float) or decimal places
        # (int). Only treat a sub-1 float as a literal step size.
        amt_prec = precision.get("amount")
        step = amt_prec if isinstance(amt_prec, float) and 0 < amt_prec < 1 else d.step_size
        return cls(
            min_notional=float(cost_min) if cost_min else d.min_notional,
            step_size=float(step),
            min_qty=float(amt_min) if amt_min else d.min_qty,
            tick_size=float(price_min) if price_min else d.tick_size,
        )


@dataclass(frozen=True)
class CostModel:
    """Per-side cost assumptions for a simulated spot fill.

    ``taker_fee`` is charged on EACH leg. ``slippage_bps`` is an additional
    adverse cost per leg, expressed in basis points of notional.
    """

    taker_fee: float = 0.0010     # 10 bps per side
    slippage_bps: float = 1.0     # 1 bp of notional per side
    filters: ExchangeFilters = ExchangeFilters()

    def fill_cost(self, notional: float) -> float:
        """Total cost (fee + slippage) charged on a single fill of ``notional``.

        One leg only. A round trip calls this twice (once per leg).
        """
        if notional <= 0:
            return 0.0
        return notional * (self.taker_fee + self.slippage_bps / 10_000.0)

    def can_trade(self, qty: float, price: float) -> bool:
        """True iff a fill of ``qty`` at ``price`` clears the exchange filters.

        A trade that fails this MUST be skipped by the engine, not resized — the
        exchange would reject it outright.
        """
        return self.filters.meets_min_notional(qty, price)

    def round_qty(self, qty: float) -> float:
        return self.filters.round_qty(qty)
