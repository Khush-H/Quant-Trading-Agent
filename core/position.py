"""Position tracking.

Tracks open positions and realized/unrealized PnL. The risk layer reads from
here for stateful checks (open-position count, exposure). Implemented during the
build; this scaffold fixes the data shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid importing the DB at module load
    from core.database import Database


@dataclass
class Position:
    """A spot holding in a single symbol.

    SPOT account: flat or long only. ``quantity`` is the size held and is
    constrained ``>= 0`` — there is no short side. Flat is ``quantity == 0``.
    This mirrors the ``positions`` table constraint in :mod:`core.database`.
    """

    symbol: str
    quantity: float = 0.0  # >= 0; long size held. Flat = 0. Never negative.
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(
                f"Spot positions cannot be short: quantity={self.quantity!r} "
                "(must be >= 0). This is a spot-only account."
            )

    @property
    def is_open(self) -> bool:
        return self.quantity > 0.0

    def unrealized_pnl(self, mark_price: float) -> float:
        """Mark-to-market PnL at the given price (long-only).

        With a long-only holding this is simply the size times the move from
        the average entry; it is zero when flat.
        """
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


class Action(str, Enum):
    """What to do to move the current holding toward a target. Spot, long-only."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class PositionDelta:
    """The change needed to reach a target spot position.

    ``action`` is buy/sell/hold; ``quantity`` is the (non-negative) size of that
    action. A hold has quantity 0. Spot, long-only throughout: both current and
    target are ``>= 0``, and we never produce a "sell" larger than the holding.
    """

    symbol: str
    current_qty: float
    target_qty: float
    action: Action
    quantity: float

    @property
    def should_buy(self) -> bool:
        return self.action is Action.BUY

    @property
    def should_sell(self) -> bool:
        return self.action is Action.SELL

    @property
    def should_hold(self) -> bool:
        return self.action is Action.HOLD


def compute_delta(
    symbol: str, current_qty: float, target_qty: float, *, tol: float = 1e-12
) -> PositionDelta:
    """Diff current vs target holding into a buy/sell/hold of a size.

    Spot, long-only: both quantities must be ``>= 0``. ``tol`` collapses
    floating-point dust to a hold so we never emit a microscopic order.
    """
    if current_qty < 0 or target_qty < 0:
        raise ValueError(
            "Spot positions are long-only: current and target quantity must be "
            f">= 0 (got current={current_qty!r}, target={target_qty!r})."
        )
    diff = target_qty - current_qty
    if abs(diff) <= tol:
        return PositionDelta(symbol, current_qty, target_qty, Action.HOLD, 0.0)
    if diff > 0:
        return PositionDelta(symbol, current_qty, target_qty, Action.BUY, diff)
    return PositionDelta(symbol, current_qty, target_qty, Action.SELL, -diff)


class PositionManager:
    """Reads/writes spot holdings from the ``positions`` table and diffs targets.

    The database is the source of truth for the current holding; this manager is
    the thin read/diff/update surface the daemon uses. Spot semantics (qty >= 0,
    no short, flat = 0) match :class:`Position` and the table CHECK constraint.
    """

    def __init__(self, db: "Database") -> None:
        self.db = db

    def current(self, symbol: str) -> Position:
        """Load the current holding as a :class:`Position` (flat if absent)."""
        row = self.db.get_position(symbol)
        if row is None:
            return Position(symbol=symbol, quantity=0.0)
        return Position(
            symbol=symbol,
            quantity=float(row["quantity"]),
            avg_entry_price=float(row["avg_entry_price"]),
            realized_pnl=float(row["realized_pnl"]),
        )

    def current_quantity(self, symbol: str) -> float:
        return self.current(symbol).quantity

    def delta_to(self, symbol: str, target_qty: float) -> PositionDelta:
        """Compute the buy/sell/hold needed to reach ``target_qty``."""
        return compute_delta(symbol, self.current_quantity(symbol), target_qty)

    def apply_fill(
        self, symbol: str, action: Action, quantity: float, price: float
    ) -> Position:
        """Update and persist the holding after a (simulated) fill.

        Long-only averaging: a buy averages into the position; a sell reduces it
        and realizes PnL on the sold portion. Quantity stays ``>= 0``.
        """
        pos = self.current(symbol)
        qty = float(quantity)
        if action is Action.BUY:
            new_qty = pos.quantity + qty
            # Weighted-average entry over the combined size.
            new_avg = (
                (pos.avg_entry_price * pos.quantity + price * qty) / new_qty
                if new_qty > 0 else 0.0
            )
            realized = pos.realized_pnl
        elif action is Action.SELL:
            qty = min(qty, pos.quantity)  # never sell more than held (spot)
            realized = pos.realized_pnl + (price - pos.avg_entry_price) * qty
            new_qty = pos.quantity - qty
            new_avg = pos.avg_entry_price if new_qty > 0 else 0.0
        else:  # HOLD
            return pos
        self.db.upsert_position(symbol, new_qty, new_avg, realized)
        return Position(
            symbol=symbol, quantity=new_qty, avg_entry_price=new_avg,
            realized_pnl=realized,
        )
