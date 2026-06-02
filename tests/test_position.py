"""Tests for the in-memory Position dataclass (spot-only).

SPOT account: a Position is flat or long, never short. These pin that the
dataclass rejects a negative quantity (matching the positions-table CHECK in
core.database) and that unrealized PnL is long-only.
"""

from __future__ import annotations

import pytest

from core.position import Position


def test_negative_quantity_rejected():
    with pytest.raises(ValueError, match="short"):
        Position(symbol="BTC/USDT", quantity=-1.0)


def test_flat_position_is_allowed_and_not_open():
    p = Position(symbol="BTC/USDT", quantity=0.0)
    assert p.is_open is False
    assert p.unrealized_pnl(mark_price=100.0) == 0.0


def test_long_position_is_open():
    p = Position(symbol="BTC/USDT", quantity=2.0)
    assert p.is_open is True


def test_unrealized_pnl_is_long_only():
    p = Position(symbol="BTC/USDT", quantity=2.0, avg_entry_price=100.0)
    # Price up 10 over a 2-unit long => +20.
    assert p.unrealized_pnl(mark_price=110.0) == pytest.approx(20.0)
    # Price down 10 => -20 (a long loses when price falls).
    assert p.unrealized_pnl(mark_price=90.0) == pytest.approx(-20.0)
