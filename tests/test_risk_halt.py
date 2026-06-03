"""Tests for the real risk gate + SYSTEM_HALT circuit breaker (PAPER only).

Covers the four required behaviours:
  (a) a -3% rolling 24h drawdown trips HALT and NO further entries are placed;
  (b) HALT does not self-clear — a new cycle while halted still refuses;
  (c) flatten-to-cash actually happens on the HALT transition (a held position
      is sold, routed through the SAME risk_check chokepoint);
  (d) an over-limit order (> 20% NAV) is rejected.

Plus the other breaker conditions (consecutive failures, stale heartbeat), the
no-bypass invariant (the flatten-sell goes through approve), and the manual
reset path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
import core.engine as engine_mod
from core.engine import Order, Side, TradingDaemon
from core.exchange import PaperExecutor
from core.database import Database
from core.position import Action, PositionManager
from core.risk import RiskEngine, rolling_drawdown_pct


HOUR_MS = 3_600_000


def _paper_settings(tmp_path, **kw):
    return Settings(
        _env_file=None, mode="paper",
        database_url=f"sqlite:///{tmp_path/'risk.db'}",
        slippage_bps=2.0, **kw,
    )


def _ohlcv(n=120, seed=5):
    """Synthetic bars; last row is the forming bar the daemon drops.

    ts = i * 1h starting at 0, so the last CLOSED bar is at (n-2) * 1h.
    """
    rng = np.random.default_rng(seed)
    ts = pd.Index(np.arange(n, dtype=np.int64) * HOUR_MS, name="ts")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    op = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(op, close) * 1.001
    low = np.minimum(op, close) * 0.999
    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close,
         "volume": rng.uniform(10, 100, n)},
        index=ts,
    )


def _daemon(tmp_path, settings, db, *, predictor, equity=100_000.0,
            position_manager=None, executor=None):
    return TradingDaemon(
        "BTC/USDT", "1h", settings=settings, db=db,
        executor=executor or PaperExecutor(settings),
        predictor=predictor, equity=equity,
        position_manager=position_manager,
    )


# --- (a) -3% rolling drawdown trips HALT; no further entries ------------------

def test_minus_3pct_drawdown_trips_halt_and_blocks_entries(tmp_path):
    settings = _paper_settings(tmp_path)          # halt_drawdown_pct = -3.0
    db = Database(settings)
    db.init_schema()

    bars = _ohlcv()
    last_closed_ts = int(bars.index[-2])          # the bar the daemon will score

    # Seed a NAV peak within the trailing 24h that is ~3.85% above the daemon's
    # current NAV (~equity 100_000, flat). Current NAV / peak - 1 <= -3%.
    peak_ts = last_closed_ts - 5 * HOUR_MS
    db.record_nav(peak_ts, 104_000.0)             # peak
    db.record_nav(last_closed_ts - HOUR_MS, 101_000.0)

    # Always-confident predictor: absent HALT this WOULD open a long.
    daemon = _daemon(tmp_path, settings, db, predictor=lambda r: 0.99)
    summary = daemon.run_once(bars)

    assert db.is_halted() is True                 # breaker tripped
    assert "drawdown" in (db.halt_reason() or "")
    # No entry was placed: the cycle resolved to a hold (flat, halted).
    assert summary["action"] == "hold"
    assert summary["accepted"] is True
    assert daemon.position_manager.current_quantity("BTC/USDT") == 0.0
    # No accepted BUY anywhere in the log.
    rows = db.recent_executions("BTC/USDT", limit=20)
    assert not any(r["action"] == "buy" and r["accepted"] == 1 for r in rows)


# --- (b) HALT does not self-clear --------------------------------------------

def test_halt_does_not_self_clear_on_next_cycle(tmp_path):
    settings = _paper_settings(tmp_path)
    db = Database(settings)
    db.init_schema()
    db.set_halt("manually tripped for test")      # pre-set HALT

    bars = _ohlcv()
    daemon = _daemon(tmp_path, settings, db, predictor=lambda r: 0.99)

    # Cycle 1 while halted.
    s1 = daemon.run_once(bars)
    assert db.is_halted() is True
    assert s1["action"] == "hold"                 # flat + halted -> hold, no buy

    # Cycle 2: still halted; the breaker condition is gone (no drawdown), yet a
    # confident long is STILL refused because HALT must not self-clear.
    s2 = daemon.run_once(bars)
    assert db.is_halted() is True
    assert s2["action"] == "hold"
    assert daemon.position_manager.current_quantity("BTC/USDT") == 0.0
    rows = db.recent_executions("BTC/USDT", limit=20)
    assert not any(r["action"] == "buy" and r["accepted"] == 1 for r in rows)

    # Only the manual reset clears it.
    db.clear_halt()
    assert db.is_halted() is False


# --- (c) flatten-to-cash on the HALT transition ------------------------------

def test_flatten_to_cash_on_halt_transition(tmp_path):
    settings = _paper_settings(tmp_path)
    db = Database(settings)
    db.init_schema()

    # Start HOLDING a position (as if a prior cycle bought).
    pm = PositionManager(db)
    pm.apply_fill("BTC/USDT", Action.BUY, 100.0, price=100.0)
    assert pm.current_quantity("BTC/USDT") == pytest.approx(100.0)

    bars = _ohlcv()
    last_closed_ts = int(bars.index[-2])

    # Seed a drawdown so HALT trips on this cycle.
    db.record_nav(last_closed_ts - 5 * HOUR_MS, 104_000.0)
    db.record_nav(last_closed_ts - HOUR_MS, 101_000.0)

    # Spy on risk_check to prove the flatten-SELL goes THROUGH the chokepoint.
    seen = []
    real = engine_mod.risk_check

    def _spy(order, **kw):
        seen.append(order.side)
        return real(order, **kw)

    daemon = _daemon(tmp_path, settings, db, predictor=lambda r: 0.99,
                     position_manager=pm)
    import pytest as _pytest
    daemon_mp = _pytest.MonkeyPatch()
    daemon_mp.setattr(engine_mod, "risk_check", _spy)
    try:
        summary = daemon.run_once(bars)
    finally:
        daemon_mp.undo()

    assert db.is_halted() is True
    # The position was flattened to cash.
    assert summary["action"] == "sell" and summary["accepted"] is True
    assert pm.current_quantity("BTC/USDT") == pytest.approx(0.0)
    # And the flatten-sell was routed through the chokepoint (no bypass).
    assert Side.SELL in seen


# --- (d) over-limit order (> 20% NAV) rejected -------------------------------

def test_over_limit_order_is_rejected(tmp_path):
    settings = _paper_settings(tmp_path)          # max_trade_fraction = 0.20
    db = Database(settings)
    db.init_schema()
    engine = RiskEngine(settings=settings, db=db)

    nav = 10_000.0
    price = 100.0
    # 21% of NAV -> 21 units * 100 = 2100 > 2000 cap.
    over = Order(symbol="BTC/USDT", side=Side.BUY, quantity=21.0, limit_price=price)
    d_over = engine.approve(over, nav=nav, current_exposure=0.0)
    assert d_over.approved is False
    assert "per-trade cap" in (d_over.reason or "")

    # 20% exactly -> allowed.
    ok = Order(symbol="BTC/USDT", side=Side.BUY, quantity=20.0, limit_price=price)
    d_ok = engine.approve(ok, nav=nav, current_exposure=0.0)
    assert d_ok.approved is True


def test_total_exposure_cap_rejects_second_position(tmp_path):
    settings = _paper_settings(tmp_path)          # max_total_exposure = 0.20
    db = Database(settings)
    db.init_schema()
    engine = RiskEngine(settings=settings, db=db)
    nav = 10_000.0
    # Already holding 20% exposure; a within-per-trade buy still breaches the
    # total-exposure cap.
    order = Order(symbol="BTC/USDT", side=Side.BUY, quantity=10.0, limit_price=100.0)
    d = engine.approve(order, nav=nav, current_exposure=2_000.0)
    assert d.approved is False
    assert "total-exposure cap" in (d.reason or "")


# --- other breaker conditions -------------------------------------------------

def test_consecutive_exchange_failures_trip_halt(tmp_path):
    settings = _paper_settings(tmp_path, halt_max_consecutive_failures=3)
    db = Database(settings)
    db.init_schema()
    engine = RiskEngine(settings=settings, db=db)
    for _ in range(3):
        db.record_exchange_failure()
    assert engine.evaluate_halt(now_ms=10 * HOUR_MS) is not None
    assert db.is_halted() is True
    assert "consecutive exchange failures" in (db.halt_reason() or "")


def test_stale_heartbeat_trips_halt(tmp_path):
    settings = _paper_settings(tmp_path, halt_heartbeat_timeout_minutes=15.0)
    db = Database(settings)
    db.init_schema()
    engine = RiskEngine(settings=settings, db=db)
    db.record_heartbeat(0)                        # heartbeat at t=0
    now = 20 * 60_000                             # 20 min later > 15 min limit
    assert engine.evaluate_halt(now_ms=now) is not None
    assert "heartbeat stale" in (db.halt_reason() or "")


def test_halt_blocks_buy_but_allows_sell(tmp_path):
    settings = _paper_settings(tmp_path)
    db = Database(settings)
    db.init_schema()
    db.set_halt("test")
    engine = RiskEngine(settings=settings, db=db)
    buy = Order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0, limit_price=100.0)
    sell = Order(symbol="BTC/USDT", side=Side.SELL, quantity=1.0, limit_price=100.0)
    assert engine.approve(buy, nav=10_000.0).approved is False
    assert engine.approve(sell, nav=10_000.0).approved is True   # flatten allowed


def test_rolling_drawdown_helper():
    hist = [[0, 100.0], [HOUR_MS, 104.0], [2 * HOUR_MS, 100.88]]
    dd = rolling_drawdown_pct(hist, now_ms=2 * HOUR_MS, window_ms=24 * HOUR_MS)
    assert dd == pytest.approx((100.88 / 104.0 - 1) * 100.0)      # ~ -3.0%
    # Too little history -> None.
    assert rolling_drawdown_pct([[0, 100.0]], now_ms=0, window_ms=HOUR_MS) is None
