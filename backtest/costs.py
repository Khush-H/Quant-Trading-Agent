"""Trading cost model.

Fees, slippage, and (optionally) funding. Backtests without realistic costs are
the most common way to fool yourself, so this is a first-class module rather
than a constant buried in the engine. Implemented in the backtest phase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-trade cost assumptions.

    Defaults are placeholders; calibrate to the target exchange's actual fee
    schedule and observed slippage before trusting any backtest.
    """

    taker_fee: float = 0.0010  # 10 bps
    maker_fee: float = 0.0002  # 2 bps
    slippage_bps: float = 1.0  # 1 bp of notional, as a simple starting model

    def cost(self, notional: float, is_taker: bool = True) -> float:
        """Estimated cost (fees + slippage) for a trade of given notional."""
        raise NotImplementedError("Cost math implemented in backtest phase.")
