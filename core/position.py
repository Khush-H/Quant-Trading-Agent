"""Position tracking.

Tracks open positions and realized/unrealized PnL. The risk layer reads from
here for stateful checks (open-position count, exposure). Implemented during the
build; this scaffold fixes the data shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Position:
    """A net position in a single symbol."""

    symbol: str
    quantity: float = 0.0  # signed: positive long, negative short
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        return self.quantity != 0.0

    def unrealized_pnl(self, mark_price: float) -> float:
        """Mark-to-market PnL at the given price."""
        return (mark_price - self.avg_entry_price) * self.quantity


class PositionBook:
    """In-memory book of positions keyed by symbol.

    The persistent source of truth is the database; this is the working view
    the engine and risk layer consult during a session. Reconciliation against
    the exchange happens during the live phase.
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    def get(self, symbol: str) -> Position:
        return self._positions.setdefault(symbol, Position(symbol=symbol))

    def open_symbols(self) -> list[str]:
        return [s for s, p in self._positions.items() if p.is_open]

    def open_count(self) -> int:
        return len(self.open_symbols())

    def apply_fill(
        self, symbol: str, side_signed_qty: float, price: float
    ) -> Position:
        """Update a position from a fill. Trading math added in the build."""
        # TODO(build): average-in, realize PnL on reductions, flip handling.
        pos = self.get(symbol)
        pos.updated_at = datetime.now(timezone.utc)
        raise NotImplementedError("Fill application implemented during build.")
