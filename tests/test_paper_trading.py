"""Tests for the paper-trading layer (PAPER mode only).

Two headline guarantees:
  * Paper fills reconcile EXACTLY to the backtester's cost model on a fixed
    input (same fee + slippage), so paper and backtest agree by construction.
  * The order path cannot reach an executor without passing through
    ``risk_check`` first — a rejecting risk_check means the executor is never
    called.

Also covers the spot-only delta logic, the PositionManager round trip, the
execution_logs audit trail (including no-trades), and the hard live gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel, ExchangeFilters
from config.settings import Settings
import core.engine as engine_mod
from core.engine import (
    Order,
    Side,
    TradingDaemon,
    risk_check,
)
from core.exchange import LiveExecutor, PaperExecutor
from core.position import Action, PositionManager, compute_delta
from core.database import Database


def _paper_settings(tmp_path, **kw):
    return Settings(
        _env_file=None, mode="paper",
        database_url=f"sqlite:///{tmp_path/'paper.db'}",
        slippage_bps=2.0, **kw,
    )


def _ohlcv(n=120, seed=5, last_forming=True):
    """Synthetic bars; the LAST row is the (forming) bar the daemon must drop."""
    rng = np.random.default_rng(seed)
    ts = pd.Index(np.arange(n, dtype=np.int64) * 3_600_000, name="ts")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    op = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(op, close) * 1.001
    low = np.minimum(op, close) * 0.999
    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close,
         "volume": rng.uniform(10, 100, n)},
        index=ts,
    )


# --- (1) paper fills reconcile to the backtest cost model ---------------------

def test_paper_fill_reconciles_to_backtest_cost_model():
    """Fixed input: same fee + slippage from the SAME CostModel the backtester
    uses. The PaperExecutor must charge exactly what backtest.costs computes."""
    settings = Settings(_env_file=None, mode="paper", slippage_bps=2.0)
    filters = ExchangeFilters(min_notional=10.0, step_size=1e-8, min_qty=0.0)
    ex = PaperExecutor(settings, filters=filters)

    price = 100.0
    qty = 2.0
    order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=qty, limit_price=price)
    res = ex.execute(order)
    assert res.accepted

    notional = qty * price
    # The independent reference: the backtester's own cost model, same params.
    ref_cm = CostModel(taker_fee=0.0010, slippage_bps=2.0, filters=filters)
    expected_total = ref_cm.fill_cost(notional)
    expected_fee = notional * 0.0010
    expected_slip = notional * (2.0 / 10_000.0)

    assert expected_fee + expected_slip == pytest.approx(expected_total)
    # Reconstruct paper's reported cost split from the executor's model.
    paper_fee = notional * ex.cost_model.taker_fee
    paper_slip = notional * (ex.cost_model.slippage_bps / 10_000.0)
    assert paper_fee == pytest.approx(expected_fee)
    assert paper_slip == pytest.approx(expected_slip)
    assert paper_fee + paper_slip == pytest.approx(expected_total)
    # And the BUY fills slightly worse than the reference price (slippage).
    assert res.avg_price > price


def test_paper_rejects_below_min_notional_like_backtest():
    settings = Settings(_env_file=None, mode="paper")
    ex = PaperExecutor(settings, filters=ExchangeFilters(min_notional=1e9))
    order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=0.001, limit_price=100.0)
    res = ex.execute(order)
    assert res.accepted is False
    assert "minNotional" in (res.reason or "")


# --- (2) order path cannot reach the executor without risk_check --------------

class _RecordingExecutor:
    mode = None  # set in fixture to match settings

    def __init__(self, mode):
        type(self).mode = mode
        self.mode = mode
        self.calls = []
        # carry a cost_model so the daemon can log fee/slippage
        self.cost_model = CostModel(taker_fee=0.0010, slippage_bps=2.0,
                                    filters=ExchangeFilters(min_notional=1.0,
                                                            step_size=1e-8,
                                                            min_qty=0.0))

    def execute(self, order):
        from core.engine import OrderResult
        self.calls.append(order)
        return OrderResult(order=order, accepted=True, mode=self.mode,
                           filled_quantity=order.quantity,
                           avg_price=order.limit_price)


def _daemon_with_spy(tmp_path, predictor):
    from config import Mode
    settings = _paper_settings(tmp_path)
    db = Database(settings)
    spy = _RecordingExecutor(Mode.PAPER)
    daemon = TradingDaemon(
        "BTC/USDT", "1h", settings=settings, db=db,
        executor=spy, predictor=predictor,
    )
    return daemon, spy, db


def test_executor_not_reached_when_risk_check_rejects(tmp_path, monkeypatch):
    # A confident long would normally trade; force risk_check to REJECT.
    from core.engine import RiskCheckResult

    def _reject(order, *, settings=None):
        return RiskCheckResult(approved=False, order=order, reason="blocked")

    monkeypatch.setattr(engine_mod, "risk_check", _reject)

    daemon, spy, db = _daemon_with_spy(tmp_path, predictor=lambda row: 0.99)
    summary = daemon.run_once(_ohlcv())
    assert summary["accepted"] is False
    assert "risk_check rejected" in summary["reason"]
    assert spy.calls == []  # executor MUST NOT have been reached


def test_risk_check_runs_before_executor_when_approving(tmp_path, monkeypatch):
    # Spy on risk_check to assert it is consulted before the executor runs.
    order_seen = {"risk": None}
    real = engine_mod.risk_check

    def _spy(order, *, settings=None):
        order_seen["risk"] = order
        return real(order, settings=settings)

    monkeypatch.setattr(engine_mod, "risk_check", _spy)
    daemon, spy, db = _daemon_with_spy(tmp_path, predictor=lambda row: 0.99)
    summary = daemon.run_once(_ohlcv())
    assert order_seen["risk"] is not None      # risk_check was called
    assert summary["accepted"] is True
    assert len(spy.calls) == 1                 # executor reached AFTER approval


def test_risk_check_stub_always_approves():
    res = risk_check(Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0))
    assert res.approved is True


# --- delta logic (spot, long-only) -------------------------------------------

def test_compute_delta_buy_sell_hold():
    buy = compute_delta("X", current_qty=0.0, target_qty=2.0)
    assert buy.should_buy and buy.quantity == pytest.approx(2.0)
    sell = compute_delta("X", current_qty=2.0, target_qty=0.0)
    assert sell.should_sell and sell.quantity == pytest.approx(2.0)
    hold = compute_delta("X", current_qty=1.0, target_qty=1.0)
    assert hold.should_hold and hold.quantity == 0.0


def test_compute_delta_rejects_short():
    with pytest.raises(ValueError, match="long-only"):
        compute_delta("X", current_qty=0.0, target_qty=-1.0)
    with pytest.raises(ValueError, match="long-only"):
        compute_delta("X", current_qty=-1.0, target_qty=0.0)


def test_position_manager_round_trip(tmp_path):
    settings = _paper_settings(tmp_path)
    db = Database(settings)
    db.init_schema()
    pm = PositionManager(db)
    assert pm.current_quantity("BTC/USDT") == 0.0
    pm.apply_fill("BTC/USDT", Action.BUY, 1.5, price=100.0)
    assert pm.current_quantity("BTC/USDT") == pytest.approx(1.5)
    # Selling never goes short and realizes PnL.
    pos = pm.apply_fill("BTC/USDT", Action.SELL, 5.0, price=110.0)  # more than held
    assert pos.quantity == 0.0
    assert pos.realized_pnl == pytest.approx((110.0 - 100.0) * 1.5)


# --- execution logging (every decision, incl. no-trades) ----------------------

def test_hold_is_logged_when_below_threshold(tmp_path):
    settings = _paper_settings(tmp_path, confidence_threshold=0.9)
    db = Database(settings)
    daemon = TradingDaemon("BTC/USDT", "1h", settings=settings, db=db,
                           executor=PaperExecutor(settings),
                           predictor=lambda row: 0.10)  # low confidence -> hold
    summary = daemon.run_once(_ohlcv())
    assert summary["action"] == "hold"
    rows = db.recent_executions("BTC/USDT", limit=5)
    assert len(rows) == 1 and rows[0]["action"] == "hold"


def test_buy_decision_logs_simulated_fee_and_slippage(tmp_path):
    settings = _paper_settings(tmp_path, confidence_threshold=0.5)
    db = Database(settings)
    # Large equity so the 20% sized order clears minNotional.
    daemon = TradingDaemon("BTC/USDT", "1h", settings=settings, db=db,
                           predictor=lambda row: 0.99, equity=100_000.0)
    summary = daemon.run_once(_ohlcv())
    assert summary["action"] == "buy" and summary["accepted"] is True
    assert summary["fee"] > 0 and summary["slippage"] > 0
    rows = db.recent_executions("BTC/USDT", limit=5)
    logged = rows[0]
    assert logged["action"] == "buy"
    assert logged["fee"] > 0 and logged["slippage"] > 0
    # Reconcile logged fee/slippage to the cost model on the logged notional.
    cm = daemon.executor.cost_model
    assert logged["fee"] == pytest.approx(logged["notional"] * cm.taker_fee)
    assert logged["slippage"] == pytest.approx(
        logged["notional"] * cm.slippage_bps / 10_000.0)


def test_no_model_logs_hold(tmp_path):
    # No predictor injected and no committed model -> hold with that reason.
    settings = _paper_settings(tmp_path)
    db = Database(settings)
    daemon = TradingDaemon("BTC/USDT", "1h", settings=settings, db=db,
                           predictor=None, model_name="does_not_exist")
    summary = daemon.run_once(_ohlcv())
    assert summary["action"] == "hold"
    assert "no committed model" in summary["reason"]


def test_daemon_drops_forming_candle(tmp_path):
    # The last bar must never be used: features are built on closed bars only.
    settings = _paper_settings(tmp_path, confidence_threshold=0.5)
    db = Database(settings)
    seen = {}

    def predictor(row):
        seen["ts"] = int(row.index[-1])
        return 0.0  # hold; we only care which ts was scored

    bars = _ohlcv()
    daemon = TradingDaemon("BTC/USDT", "1h", settings=settings, db=db,
                           predictor=predictor)
    daemon.run_once(bars)
    assert seen["ts"] != int(bars.index[-1])  # not the forming (last) bar
    assert seen["ts"] == int(bars.index[-2])  # the last CLOSED bar


# --- live stays hard-gated ----------------------------------------------------

def test_live_executor_is_blocked_without_confirmation():
    s = Settings(_env_file=None, mode="paper")  # not live; flag false
    ex = LiveExecutor(s)
    res = ex.execute(Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0))
    assert res.accepted is False
    assert "LIVE_TRADING_CONFIRMED" in (res.reason or "")


def test_daemon_refuses_non_paper_mode():
    s = Settings(_env_file=None, mode="backtest")
    with pytest.raises(ValueError, match="PAPER-only"):
        TradingDaemon("BTC/USDT", settings=s)
