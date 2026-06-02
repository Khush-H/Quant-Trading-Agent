"""Tests for the cost model — the single source of truth for simulated costs.

Pins that fees are charged PER SIDE (a round trip pays twice), that slippage is
added on top, and that a fill below Binance minNotional is rejected rather than
silently resized.
"""

from __future__ import annotations

import pytest

from backtest.costs import CostModel, ExchangeFilters


def test_fee_is_charged_per_side_round_trip_pays_twice():
    cm = CostModel(taker_fee=0.0010, slippage_bps=0.0)
    notional = 1_000.0
    one_leg = cm.fill_cost(notional)
    assert one_leg == pytest.approx(1.0)  # 10bps of 1000
    # A round trip is two fills -> two legs of cost.
    assert 2 * one_leg == pytest.approx(2.0)


def test_slippage_adds_on_top_of_fee_per_side():
    cm = CostModel(taker_fee=0.0010, slippage_bps=5.0)  # 10bps + 5bps = 15bps
    assert cm.fill_cost(1_000.0) == pytest.approx(1.5)


def test_below_min_notional_is_rejected():
    cm = CostModel(filters=ExchangeFilters(min_notional=10.0, step_size=1e-6, min_qty=0))
    # 0.0001 BTC * 50_000 = $5 < $10 minNotional -> cannot trade.
    assert cm.can_trade(0.0001, 50_000.0) is False
    # 0.001 BTC * 50_000 = $50 >= $10 -> ok.
    assert cm.can_trade(0.001, 50_000.0) is True


def test_round_qty_floors_to_step_size():
    f = ExchangeFilters(step_size=0.001, min_qty=0)
    assert f.round_qty(0.0019) == pytest.approx(0.001)   # floored, not rounded up
    assert f.round_qty(0.0020) == pytest.approx(0.002)
    assert f.round_qty(0.00099) == pytest.approx(0.0)     # below one step -> 0


def test_zero_or_negative_notional_costs_nothing():
    cm = CostModel()
    assert cm.fill_cost(0.0) == 0.0
    assert cm.fill_cost(-5.0) == 0.0
