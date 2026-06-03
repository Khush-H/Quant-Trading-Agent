"""Tests for the read-only web dashboard (PAPER scope).

Two safety properties are locked in here:

  1. Every DB query a data endpoint issues is read-only. The dashboard's data
     request path must touch NO write method — no INSERT/UPDATE/DELETE/REPLACE,
     and the connection it uses is opened read-only so the engine itself would
     reject a write.

  2. The ONLY state-changing endpoint — reset-halt — requires its guard. Without
     an explicit confirmation it clears nothing; with it, it clears SYSTEM_HALT
     through the SAME ``clear_halt`` mechanism ``scripts/reset_halt.py`` uses.

The endpoints are exercised by calling the route functions directly (no httpx /
TestClient dependency), with ``web.app.get_settings`` / ``get_reader`` pointed at
a temp database.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3

import pytest

from config.settings import Settings
from core.database import Database
import web.app as web_app
from web.data import DashboardData


HOUR_MS = 3_600_000
_WRITE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|TRUNCATE)\b",
    re.IGNORECASE,
)


def _settings(tmp_path, **kw) -> Settings:
    return Settings(
        _env_file=None, mode="paper",
        database_url=f"sqlite:///{tmp_path/'dash.db'}", **kw,
    )


def _seed(db: Database, *, now_ms: int, halted: bool = False) -> None:
    """Populate a realistic snapshot: a position, executions, NAV, heartbeat."""
    db.init_schema()
    db.upsert_position("BTC/USDT", quantity=1.5, avg_entry_price=100.0,
                       realized_pnl=12.5)
    db.log_execution(mode="paper", symbol="BTC/USDT", timeframe="1h",
                     action="buy", ts=now_ms - 2 * HOUR_MS, price=100.0,
                     quantity=1.5, notional=150.0, fee=0.15, slippage=0.03,
                     accepted=True, reason="paper fill (simulated)")
    db.log_execution(mode="paper", symbol="BTC/USDT", timeframe="1h",
                     action="hold", ts=now_ms - HOUR_MS, price=101.0,
                     accepted=True, reason="target matches holding")
    db.record_nav(now_ms - 23 * HOUR_MS, 10_000.0)
    db.record_nav(now_ms - HOUR_MS, 10_150.0)
    db.record_heartbeat(now_ms - HOUR_MS)
    if halted:
        db.set_halt("rolling 24h drawdown -3.20% <= -3.00% threshold")


def _make_request(method: str, *, query: bytes = b"", body: bytes = b""):
    """Build a minimal Starlette Request with an optional urlencoded body."""
    from starlette.requests import Request

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http", "method": method, "headers": [],
        "query_string": query, "path": "/",
    }
    return Request(scope, receive=_receive)


# --- (1) data endpoints perform NO writes ------------------------------------

class _SpyConnection:
    """Wraps a real sqlite3 connection and records every SQL statement run."""

    def __init__(self, conn: sqlite3.Connection, log: list[str]) -> None:
        self._conn = conn
        self._log = log

    def execute(self, sql, *args, **kwargs):
        self._log.append(sql)
        return self._conn.execute(sql, *args, **kwargs)

    def executemany(self, sql, *args, **kwargs):
        self._log.append(sql)
        return self._conn.executemany(sql, *args, **kwargs)

    def executescript(self, sql, *args, **kwargs):
        self._log.append(sql)
        return self._conn.executescript(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)


def _spying_reader(statements: list[str], settings) -> DashboardData:
    """A DashboardData whose connections record all SQL into ``statements``."""
    reader = DashboardData(settings)
    real_connect = reader._connect_ro

    def _connect_ro():
        return _SpyConnection(real_connect(), statements)

    reader._connect_ro = _connect_ro  # type: ignore[method-assign]
    return reader


def _assert_all_reads(statements: list[str]) -> None:
    assert statements, "expected the data path to issue SQL"
    offenders = [s for s in statements if _WRITE_RE.search(s)]
    assert offenders == [], f"data path issued write SQL: {offenders!r}"
    for s in statements:
        head = s.strip().split(None, 1)[0].upper()
        assert head in {"SELECT", "PRAGMA"}, f"non-read statement: {s!r}"


def test_snapshot_issues_only_read_queries(tmp_path):
    now = 100 * HOUR_MS
    settings = _settings(tmp_path)
    _seed(Database(settings), now_ms=now)

    statements: list[str] = []
    reader = _spying_reader(statements, settings)
    snap = reader.snapshot(now_ms=now)

    # The snapshot actually read state (sanity: not an empty no-op).
    assert snap["positions"] and snap["executions"]
    _assert_all_reads(statements)


def test_api_state_endpoint_does_not_write(tmp_path, monkeypatch):
    now = 100 * HOUR_MS
    settings = _settings(tmp_path)
    _seed(Database(settings), now_ms=now)

    statements: list[str] = []
    reader = _spying_reader(statements, settings)
    monkeypatch.setattr(web_app, "get_reader", lambda: reader)

    resp = web_app.api_state()
    assert resp.status_code == 200
    _assert_all_reads(statements)


def test_index_endpoint_does_not_write(tmp_path, monkeypatch):
    now = 100 * HOUR_MS
    settings = _settings(tmp_path)
    _seed(Database(settings), now_ms=now, halted=True)

    statements: list[str] = []
    reader = _spying_reader(statements, settings)
    monkeypatch.setattr(web_app, "get_reader", lambda: reader)

    resp = web_app.index(_make_request("GET"))
    assert resp.status_code == 200
    body = resp.body.decode()
    assert "SYSTEM HALT" in body and "Reset HALT" in body  # halted UI rendered
    _assert_all_reads(statements)


def test_reason_text_is_html_escaped(tmp_path, monkeypatch):
    """Free-text from the DB (the execution reason) must be escaped, never
    rendered as raw markup."""
    now = 100 * HOUR_MS
    settings = _settings(tmp_path)
    db = Database(settings)
    db.init_schema()
    db.log_execution(mode="paper", symbol="BTC/USDT", timeframe="1h",
                     action="hold", ts=now, price=100.0, accepted=True,
                     reason="<script>alert('x')</script>")
    db.record_heartbeat(now)

    monkeypatch.setattr(web_app, "get_reader", lambda: DashboardData(settings))
    body = web_app.index(_make_request("GET")).body.decode()
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body


def test_readonly_connection_rejects_writes(tmp_path):
    """Defence-in-depth: the dashboard's connection is opened read-only, so the
    SQLite engine itself refuses any write that slipped through."""
    settings = _settings(tmp_path)
    _seed(Database(settings), now_ms=10 * HOUR_MS)

    reader = DashboardData(settings)
    conn = reader._connect_ro()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("UPDATE system_state SET value='0' WHERE key='SYSTEM_HALT';")
    finally:
        conn.close()


# --- (2) the reset endpoint requires its guard -------------------------------

def _post_reset(*, body: bytes = b"", query: bytes = b""):
    req = _make_request("POST", body=body, query=query)
    return asyncio.run(web_app.reset_halt(req))


def test_reset_halt_requires_confirm_guard(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    db = Database(settings)
    _seed(db, now_ms=10 * HOUR_MS, halted=True)
    monkeypatch.setattr(web_app, "get_settings", lambda: settings)

    assert db.is_halted() is True
    # No confirmation -> the guard refuses; HALT is NOT cleared.
    resp = _post_reset(body=b"")
    assert resp.status_code == 303
    assert "missing_confirm" in resp.headers["location"]
    assert db.is_halted() is True, "HALT must NOT clear without the guard"

    # A falsey confirm value is also refused.
    resp2 = _post_reset(body=b"confirm=no")
    assert "missing_confirm" in resp2.headers["location"]
    assert db.is_halted() is True


def test_reset_halt_clears_when_confirmed(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    db = Database(settings)
    _seed(db, now_ms=10 * HOUR_MS, halted=True)
    db.record_exchange_failure()
    db.record_exchange_failure()
    monkeypatch.setattr(web_app, "get_settings", lambda: settings)

    assert db.is_halted() is True
    resp = _post_reset(body=b"confirm=true")
    assert resp.status_code == 303
    assert "cleared" in resp.headers["location"]
    # Cleared through the SAME mechanism as scripts/reset_halt.py.
    assert db.is_halted() is False
    assert db.exchange_consecutive_failures() == 0


def test_reset_halt_routes_through_clear_halt(tmp_path, monkeypatch):
    """The confirmed reset must call Database.clear_halt — the one guarded path —
    not some second, ad-hoc way of flipping the flag."""
    settings = _settings(tmp_path)
    db = Database(settings)
    _seed(db, now_ms=10 * HOUR_MS, halted=True)
    monkeypatch.setattr(web_app, "get_settings", lambda: settings)

    calls = {"clear_halt": 0, "reset_fails": 0}
    real_clear = Database.clear_halt
    real_reset = Database.reset_exchange_failures

    def _clear(self):
        calls["clear_halt"] += 1
        return real_clear(self)

    def _reset(self):
        calls["reset_fails"] += 1
        return real_reset(self)

    monkeypatch.setattr(Database, "clear_halt", _clear)
    monkeypatch.setattr(Database, "reset_exchange_failures", _reset)

    _post_reset(body=b"confirm=yes")
    assert calls["clear_halt"] == 1
    assert calls["reset_fails"] == 1


def test_reset_halt_noop_when_not_halted(tmp_path, monkeypatch):
    """Mirrors the script's 'nothing to do' guard: a confirmed reset on a system
    that isn't halted clears nothing and reports not_halted."""
    settings = _settings(tmp_path)
    db = Database(settings)
    _seed(db, now_ms=10 * HOUR_MS, halted=False)
    monkeypatch.setattr(web_app, "get_settings", lambda: settings)

    assert db.is_halted() is False
    resp = _post_reset(body=b"confirm=true")
    assert resp.status_code == 303
    assert "not_halted" in resp.headers["location"]
